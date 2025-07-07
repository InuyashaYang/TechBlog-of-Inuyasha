这里粘贴一下OpenAI在https://github.com/openai/sparse_autoencoder库中的train.py源代码，这段代码里面的料非常多，不过不便于迅速理解并进行部署，我们会慢慢学习并解读：

```python
# bare bones training script using sparse kernels and sharding/data parallel.
# the main purpose of this code is to provide a reference implementation to compare
# against when implementing our training methodology into other codebases, and to
# demonstrate how sharding/DP can be implemented for autoencoders. some limitations:
# - many basic features (e.g checkpointing, data loading, validation) are not implemented,
# - the codebase is not designed to be extensible or easily hackable.
# - this code is not guaranteed to run efficiently out of the box / in
#   combination with other changes, so you should profile it and make changes as needed.
#
# example launch command:
#    torchrun --nproc-per-node 8 train.py


import os
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from sparse_autoencoder.kernels import *
from torch.distributed import ReduceOp

RANK = int(os.environ.get("RANK", "0"))


## parallelism


@dataclass
class Comm:
    group: torch.distributed.ProcessGroup

    def all_reduce(self, x, op=ReduceOp.SUM, async_op=False):
        return dist.all_reduce(x, op=op, group=self.group, async_op=async_op)

    def all_gather(self, x_list, x, async_op=False):
        return dist.all_gather(list(x_list), x, group=self.group, async_op=async_op)

    def broadcast(self, x, src, async_op=False):
        return dist.broadcast(x, src, group=self.group, async_op=async_op)

    def barrier(self):
        return dist.barrier(group=self.group)

    def size(self):
        return self.group.size()


@dataclass
class ShardingComms:
    n_replicas: int
    n_op_shards: int
    dp_rank: int
    sh_rank: int
    dp_comm: Comm | None
    sh_comm: Comm | None
    _rank: int

    def sh_allreduce_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sh_comm is None:
            return x

        class AllreduceForward(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                assert self.sh_comm is not None
                self.sh_comm.all_reduce(input, async_op=True)
                return input

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output

        return AllreduceForward.apply(x)  # type: ignore

    def sh_allreduce_backward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sh_comm is None:
            return x

        class AllreduceBackward(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                return input

            @staticmethod
            def backward(ctx, grad_output):
                grad_output = grad_output.clone()
                assert self.sh_comm is not None
                self.sh_comm.all_reduce(grad_output, async_op=True)
                return grad_output

        return AllreduceBackward.apply(x)  # type: ignore

    def init_broadcast_(self, autoencoder):
        if self.dp_comm is not None:
            for p in autoencoder.parameters():
                self.dp_comm.broadcast(
                    maybe_transpose(p.data),
                    replica_shard_to_rank(
                        replica_idx=0,
                        shard_idx=self.sh_rank,
                        n_op_shards=self.n_op_shards,
                    ),
                )
        
        if self.sh_comm is not None:
            # pre_bias is the same across all shards
            self.sh_comm.broadcast(
                autoencoder.pre_bias.data,
                replica_shard_to_rank(
                    replica_idx=self.dp_rank,
                    shard_idx=0,
                    n_op_shards=self.n_op_shards,
                ),
            )

    def dp_allreduce_(self, autoencoder) -> None:
        if self.dp_comm is None:
            return

        for param in autoencoder.parameters():
            if param.grad is not None:
                self.dp_comm.all_reduce(maybe_transpose(param.grad), op=ReduceOp.AVG, async_op=True)

        # make sure statistics for dead neurons are correct
        self.dp_comm.all_reduce(  # type: ignore
            autoencoder.stats_last_nonzero, op=ReduceOp.MIN, async_op=True
        )

    def sh_allreduce_scale(self, scaler):
        if self.sh_comm is None:
            return

        if hasattr(scaler, "_scale") and scaler._scale is not None:
            self.sh_comm.all_reduce(scaler._scale, op=ReduceOp.MIN, async_op=True)
            self.sh_comm.all_reduce(scaler._growth_tracker, op=ReduceOp.MIN, async_op=True)

    def _sh_comm_op(self, x, op):
        if isinstance(x, (float, int)):
            x = torch.tensor(x, device="cuda")

        if not x.is_cuda:
            x = x.cuda()

        if self.sh_comm is None:
            return x

        out = x.clone()
        self.sh_comm.all_reduce(x, op=op, async_op=True)
        return out

    def sh_sum(self, x: torch.Tensor) -> torch.Tensor:
        return self._sh_comm_op(x, ReduceOp.SUM)

    def all_broadcast(self, x: torch.Tensor) -> torch.Tensor:
        if self.dp_comm is not None:
            self.dp_comm.broadcast(
                x,
                replica_shard_to_rank(
                    replica_idx=0,
                    shard_idx=self.sh_rank,
                    n_op_shards=self.n_op_shards,
                ),
            )

        if self.sh_comm is not None:
            self.sh_comm.broadcast(
                x,
                replica_shard_to_rank(
                    replica_idx=self.dp_rank,
                    shard_idx=0,
                    n_op_shards=self.n_op_shards,
                ),
            )

        return x


def make_torch_comms(n_op_shards=4, n_replicas=2):
    if "RANK" not in os.environ:
        assert n_op_shards == 1
        assert n_replicas == 1
        return TRIVIAL_COMMS

    rank = int(os.environ.get("RANK"))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % 8)

    print(f"{rank=}, {world_size=}")
    dist.init_process_group("nccl")

    my_op_shard_idx = rank % n_op_shards
    my_replica_idx = rank // n_op_shards

    shard_rank_lists = [list(range(i, i + n_op_shards)) for i in range(0, world_size, n_op_shards)]

    shard_groups = [dist.new_group(shard_rank_list) for shard_rank_list in shard_rank_lists]

    my_shard_group = shard_groups[my_replica_idx]

    replica_rank_lists = [
        list(range(i, n_op_shards * n_replicas, n_op_shards)) for i in range(n_op_shards)
    ]

    replica_groups = [dist.new_group(replica_rank_list) for replica_rank_list in replica_rank_lists]

    my_replica_group = replica_groups[my_op_shard_idx]

    torch.distributed.all_reduce(torch.ones(1).cuda())
    torch.cuda.synchronize()

    dp_comm = Comm(group=my_replica_group)
    sh_comm = Comm(group=my_shard_group)

    return ShardingComms(
        n_replicas=n_replicas,
        n_op_shards=n_op_shards,
        dp_comm=dp_comm,
        sh_comm=sh_comm,
        dp_rank=my_replica_idx,
        sh_rank=my_op_shard_idx,
        _rank=rank,
    )


def replica_shard_to_rank(replica_idx, shard_idx, n_op_shards):
    return replica_idx * n_op_shards + shard_idx


TRIVIAL_COMMS = ShardingComms(
    n_replicas=1,
    n_op_shards=1,
    dp_rank=0,
    sh_rank=0,
    dp_comm=None,
    sh_comm=None,
    _rank=0,
)


def sharded_topk(x, k, sh_comm, capacity_factor=None):
    batch = x.shape[0]

    if capacity_factor is not None:
        k_in = min(int(k * capacity_factor // sh_comm.size()), k)
    else:
        k_in = k

    topk = torch.topk(x, k=k_in, dim=-1)
    inds = topk.indices
    vals = topk.values

    if sh_comm is None:
        return inds, vals

    all_vals = torch.empty(sh_comm.size(), batch, k_in, dtype=vals.dtype, device=vals.device)
    sh_comm.all_gather(all_vals, vals, async_op=True)

    all_vals = all_vals.permute(1, 0, 2)  # put shard dim next to k
    all_vals = all_vals.reshape(batch, -1)  # flatten shard into k

    all_topk = torch.topk(all_vals, k=k, dim=-1)
    global_topk = all_topk.values

    dummy_vals = torch.zeros_like(vals)
    dummy_inds = torch.zeros_like(inds)

    my_inds = torch.where(vals >= global_topk[:, [-1]], inds, dummy_inds)
    my_vals = torch.where(vals >= global_topk[:, [-1]], vals, dummy_vals)

    return my_inds, my_vals


## autoencoder


class FastAutoencoder(nn.Module):
    """
    Top-K Autoencoder with sparse kernels. Implements:

        latents = relu(topk(encoder(x - pre_bias) + latent_bias))
        recons = decoder(latents) + pre_bias
    """

    def __init__(
        self,
        n_dirs_local: int,
        d_model: int,
        k: int,
        auxk: int | None,
        dead_steps_threshold: int,
        comms: ShardingComms | None = None,
    ):
        super().__init__()
        self.n_dirs_local = n_dirs_local
        self.d_model = d_model
        self.k = k
        self.auxk = auxk
        self.comms = comms if comms is not None else TRIVIAL_COMMS
        self.dead_steps_threshold = dead_steps_threshold

        self.encoder = nn.Linear(d_model, n_dirs_local, bias=False)
        self.decoder = nn.Linear(n_dirs_local, d_model, bias=False)

        self.pre_bias = nn.Parameter(torch.zeros(d_model))
        self.latent_bias = nn.Parameter(torch.zeros(n_dirs_local))

        self.stats_last_nonzero: torch.Tensor
        self.register_buffer("stats_last_nonzero", torch.zeros(n_dirs_local, dtype=torch.long))

        def auxk_mask_fn(x):
            dead_mask = self.stats_last_nonzero > dead_steps_threshold
            x.data *= dead_mask  # inplace to save memory
            return x

        self.auxk_mask_fn = auxk_mask_fn

        ## initialization

        # "tied" init
        self.decoder.weight.data = self.encoder.weight.data.T.clone()

        # store decoder in column major layout for kernel
        self.decoder.weight.data = self.decoder.weight.data.T.contiguous().T

        unit_norm_decoder_(self)

    @property
    def n_dirs(self):
        return self.n_dirs_local * self.comms.n_op_shards

    def forward(self, x):
        class EncWrapper(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, pre_bias, weight, latent_bias):
                x = x - pre_bias
                latents_pre_act = F.linear(x, weight, latent_bias)

                inds, vals = sharded_topk(
                    latents_pre_act,
                    k=self.k,
                    sh_comm=self.comms.sh_comm,
                    capacity_factor=4,
                )

                ## set num nonzero stat ##
                tmp = torch.zeros_like(self.stats_last_nonzero)
                tmp.scatter_add_(
                    0,
                    inds.reshape(-1),
                    (vals > 1e-3).to(tmp.dtype).reshape(-1),
                )
                self.stats_last_nonzero *= 1 - tmp.clamp(max=1)
                self.stats_last_nonzero += 1
                ## end stats ##

                ## auxk
                if self.auxk is not None:  # for auxk
                    # IMPORTANT: has to go after stats update!
                    # WARN: auxk_mask_fn can mutate latents_pre_act!
                    auxk_inds, auxk_vals = sharded_topk(
                        self.auxk_mask_fn(latents_pre_act),
                        k=self.auxk,
                        sh_comm=self.comms.sh_comm,
                        capacity_factor=2,
                    )
                    ctx.save_for_backward(x, weight, inds, auxk_inds)
                else:
                    ctx.save_for_backward(x, weight, inds)
                    auxk_inds = None
                    auxk_vals = None

                ## end auxk

                return (
                    inds,
                    vals,
                    auxk_inds,
                    auxk_vals,
                )

            @staticmethod
            def backward(ctx, _, grad_vals, __, grad_auxk_vals):
                # encoder backwards
                if self.auxk is not None:
                    x, weight, inds, auxk_inds = ctx.saved_tensors

                    all_inds = torch.cat((inds, auxk_inds), dim=-1)
                    all_grad_vals = torch.cat((grad_vals, grad_auxk_vals), dim=-1)
                else:
                    x, weight, inds = ctx.saved_tensors

                    all_inds = inds
                    all_grad_vals = grad_vals

                grad_sum = torch.zeros(self.n_dirs_local, dtype=torch.float32, device=grad_vals.device)
                grad_sum.scatter_add_(
                    -1, all_inds.flatten(), all_grad_vals.flatten().to(torch.float32)
                )

                return (
                    None,
                    # pre_bias grad optimization - can reduce before mat-vec multiply
                    -(grad_sum @ weight),
                    triton_sparse_transpose_dense_matmul(all_inds, all_grad_vals, x, N=self.n_dirs_local),
                    grad_sum,
                )

        pre_bias = self.comms.sh_allreduce_backward(self.pre_bias)

        # encoder
        inds, vals, auxk_inds, auxk_vals = EncWrapper.apply(
            x, pre_bias, self.encoder.weight, self.latent_bias
        )

        vals = torch.relu(vals)
        if auxk_vals is not None:
            auxk_vals = torch.relu(auxk_vals)

        recons = self.decode_sparse(inds, vals)

        return recons, {
            "auxk_inds": auxk_inds,
            "auxk_vals": auxk_vals,
        }

    def decode_sparse(self, inds, vals):
        recons = TritonDecoderAutograd.apply(inds, vals, self.decoder.weight)
        recons = self.comms.sh_allreduce_forward(recons)

        return recons + self.pre_bias


def unit_norm_decoder_(autoencoder: FastAutoencoder) -> None:
    """
    Unit normalize the decoder weights of an autoencoder.
    """
    autoencoder.decoder.weight.data /= autoencoder.decoder.weight.data.norm(dim=0)


def unit_norm_decoder_grad_adjustment_(autoencoder) -> None:
    """project out gradient information parallel to the dictionary vectors - assumes that the decoder is already unit normed"""

    assert autoencoder.decoder.weight.grad is not None

    triton_add_mul_(
        autoencoder.decoder.weight.grad,
        torch.einsum("bn,bn->n", autoencoder.decoder.weight.data, autoencoder.decoder.weight.grad),
        autoencoder.decoder.weight.data,
        c=-1,
    )


def maybe_transpose(x):
    return x.T if not x.is_contiguous() and x.T.is_contiguous() else x


def sharded_grad_norm(autoencoder, comms, exclude=None):
    if exclude is None:
        exclude = []
    total_sq_norm = torch.zeros((), device="cuda", dtype=torch.float32)
    exclude = set(exclude)

    total_num_params = 0
    for param in autoencoder.parameters():
        if param in exclude:
            continue
        if param.grad is not None:
            sq_norm = ((param.grad).float() ** 2).sum()
            if param is autoencoder.pre_bias:
                total_sq_norm += sq_norm  # pre_bias is the same across all shards
            else:
                total_sq_norm += comms.sh_sum(sq_norm)

            param_shards = comms.n_op_shards if param is autoencoder.pre_bias else 1
            total_num_params += param.numel() * param_shards

    return total_sq_norm.sqrt()


def batch_tensors(
    it: Iterable[torch.Tensor],
    batch_size: int,
    drop_last=True,
    stream=None,
) -> Iterator[torch.Tensor]:
    """
    input is iterable of tensors of shape [batch_old, ...]
    output is iterable of tensors of shape [batch_size, ...]
    batch_old does not need to be divisible by batch_size
    """

    tensors = []
    batch_so_far = 0

    for t in it:
        tensors.append(t)
        batch_so_far += t.shape[0]

        if sum(t.shape[0] for t in tensors) < batch_size:
            continue

        while batch_so_far >= batch_size:
            if len(tensors) == 1:
                (concat,) = tensors
            else:
                with torch.cuda.stream(stream):
                    concat = torch.cat(tensors, dim=0)

            offset = 0
            while offset + batch_size <= concat.shape[0]:
                yield concat[offset : offset + batch_size]
                batch_so_far -= batch_size
                offset += batch_size

            tensors = [concat[offset:]] if offset < concat.shape[0] else []

    if len(tensors) > 0 and not drop_last:
        yield torch.cat(tensors, dim=0)


def print0(*a, **k):
    if RANK == 0:
        print(*a, **k)


import wandb


class Logger:
    def __init__(self, **kws):
        self.vals = {}
        self.enabled = (RANK == 0) and not kws.pop("dummy", False)
        if self.enabled:
            wandb.init(
                **kws
            )

    def logkv(self, k, v):
        if self.enabled:
            self.vals[k] = v.detach() if isinstance(v, torch.Tensor) else v
        return v

    def dumpkvs(self):
        if self.enabled:
            wandb.log(self.vals)
            self.vals = {}


def training_loop_(
    ae, train_acts_iter, loss_fn, lr, comms, eps=6.25e-10, clip_grad=None, ema_multiplier=0.999, logger=None
):
    if logger is None:
        logger = Logger(dummy=True)

    scaler = torch.cuda.amp.GradScaler()
    autocast_ctx_manager = torch.cuda.amp.autocast()

    opt = torch.optim.Adam(ae.parameters(), lr=lr, eps=eps, fused=True)
    if ema_multiplier is not None:
        ema = EmaModel(ae, ema_multiplier=ema_multiplier)

    for i, flat_acts_train_batch in enumerate(train_acts_iter):
        flat_acts_train_batch = flat_acts_train_batch.cuda()

        with autocast_ctx_manager:
            recons, info = ae(flat_acts_train_batch)

            loss = loss_fn(ae, flat_acts_train_batch, recons, info, logger)

        print0(i, loss)

        logger.logkv("loss_scale", scaler.get_scale())

        if RANK == 0:
            wandb.log({"train_loss": loss.item()})

        loss = scaler.scale(loss)
        loss.backward()

        unit_norm_decoder_(ae)
        unit_norm_decoder_grad_adjustment_(ae)

        # allreduce gradients
        comms.dp_allreduce_(ae)

        # keep fp16 loss scale synchronized across shards
        comms.sh_allreduce_scale(scaler)

        # if you want to do anything with the gradients that depends on the absolute scale (e.g clipping, do it after the unscale_)
        scaler.unscale_(opt)

        # gradient clipping
        if clip_grad is not None:
            grad_norm = sharded_grad_norm(ae, comms)
            logger.logkv("grad_norm", grad_norm)
            grads = [x.grad for x in ae.parameters() if x.grad is not None]
            torch._foreach_mul_(grads, clip_grad / torch.clamp(grad_norm, min=clip_grad))

        if ema_multiplier is not None:
            ema.step()

        # take step with optimizer
        scaler.step(opt)
        scaler.update()
        
        logger.dumpkvs()


def init_from_data_(ae, stats_acts_sample, comms):
    from geom_median.torch import compute_geometric_median

    ae.pre_bias.data = (
        compute_geometric_median(stats_acts_sample[:32768].float().cpu()).median.cuda().float()
    )
    comms.all_broadcast(ae.pre_bias.data)

    # encoder initialization (note: in our ablations we couldn't find clear evidence that this is beneficial, this is just to ensure exact match with internal codebase)
    d_model = ae.d_model
    with torch.no_grad():
        x = torch.randn(256, d_model).cuda().to(stats_acts_sample.dtype)
        x /= x.norm(dim=-1, keepdim=True)
        x += ae.pre_bias.data
        comms.all_broadcast(x)
        recons, _ = ae(x)
        recons_norm = (recons - ae.pre_bias.data).norm(dim=-1).mean()

        ae.encoder.weight.data /= recons_norm.item()
        print0("x norm", x.norm(dim=-1).mean().item())
        print0("out norm", (ae(x)[0] - ae.pre_bias.data).norm(dim=-1).mean().item())


from contextlib import contextmanager


@contextmanager
def temporary_weight_swap(model: torch.nn.Module, new_weights: list[torch.Tensor]):
    for _p, new_p in zip(model.parameters(), new_weights, strict=True):
        assert _p.shape == new_p.shape
        _p.data, new_p.data = new_p.data, _p.data

    yield

    for _p, new_p in zip(model.parameters(), new_weights, strict=True):
        assert _p.shape == new_p.shape
        _p.data, new_p.data = new_p.data, _p.data


class EmaModel:
    def __init__(self, model, ema_multiplier):
        self.model = model
        self.ema_multiplier = ema_multiplier
        self.ema_weights = [torch.zeros_like(x, requires_grad=False) for x in model.parameters()]
        self.ema_steps = 0

    def step(self):
        torch._foreach_lerp_(
            self.ema_weights,
            list(self.model.parameters()),
            1 - self.ema_multiplier,
        )
        self.ema_steps += 1

    # context manager for setting the autoencoder weights to the EMA weights
    @contextmanager
    def use_ema_weights(self):
        assert self.ema_steps > 0

        # apply bias correction
        bias_correction = 1 - self.ema_multiplier**self.ema_steps
        ema_weights_bias_corrected = torch._foreach_div(self.ema_weights, bias_correction)

        with torch.no_grad():
            with temporary_weight_swap(self.model, ema_weights_bias_corrected):
                yield


@dataclass
class Config:
    n_op_shards: int = 1
    n_replicas: int = 8

    n_dirs: int = 32768
    bs: int = 131072
    d_model: int = 768
    k: int = 32
    auxk: int = 256

    lr: float = 1e-4
    eps: float = 6.25e-10
    clip_grad: float | None = None
    auxk_coef: float = 1 / 32
    dead_toks_threshold: int = 10_000_000
    ema_multiplier: float | None = None
    
    wandb_project: str | None = None
    wandb_name: str | None = None


def main():
    cfg = Config()
    comms = make_torch_comms(n_op_shards=cfg.n_op_shards, n_replicas=cfg.n_replicas)

    ## dataloading is left as an exercise for the reader
    acts_iter = ...
    stats_acts_sample = ...

    n_dirs_local = cfg.n_dirs // cfg.n_op_shards
    bs_local = cfg.bs // cfg.n_replicas

    ae = FastAutoencoder(
        n_dirs_local=n_dirs_local,
        d_model=cfg.d_model,
        k=cfg.k,
        auxk=cfg.auxk,
        dead_steps_threshold=cfg.dead_toks_threshold // cfg.bs,
        comms=comms,
    )
    ae.cuda()
    init_from_data_(ae, stats_acts_sample, comms)
    # IMPORTANT: make sure all DP ranks have the same params
    comms.init_broadcast_(ae)

    mse_scale = (
        1 / ((stats_acts_sample.float().mean(dim=0) - stats_acts_sample.float()) ** 2).mean()
    )
    comms.all_broadcast(mse_scale)
    mse_scale = mse_scale.item()

    logger = Logger(
        project=cfg.wandb_project,
        name=cfg.wandb_name,
        dummy=cfg.wandb_project is None,
    )

    training_loop_(
        ae,
        batch_tensors(
            acts_iter,
            bs_local,
            drop_last=True,
        ),
        lambda ae, flat_acts_train_batch, recons, info, logger: (
            # MSE
            logger.logkv("train_recons", mse_scale * mse(recons, flat_acts_train_batch))
            # AuxK
            + logger.logkv(
                "train_maxk_recons",
                cfg.auxk_coef
                * normalized_mse(
                    ae.decode_sparse(
                        info["auxk_inds"],
                        info["auxk_vals"],
                    ),
                    flat_acts_train_batch - recons.detach() + ae.pre_bias.detach(),
                ).nan_to_num(0),
            )
        ),
        lr=cfg.lr,
        eps=cfg.eps,
        clip_grad=cfg.clip_grad,
        ema_multiplier=cfg.ema_multiplier,
        logger=logger,
        comms=comms,
    )


if __name__ == "__main__":
    main()
```
我们创建了一个简化版的训练代码，抛弃了其中所有的分布式运行、高效训练和模型平滑操作，并补充了其中的数据训练类：
```python
# sparse_autoencoder/my_train.py

import os
import torch
import torch.nn as nn
import h5py
import wandb
from dataclasses import dataclass
from typing import Iterable, Iterator
from tqdm import tqdm

# 关键修改：使用相对导入，从当前包中导入其他模块
# 假设 FastAutoencoder 等核心逻辑已经包含在本文件中
# 如果需要从其他文件导入，例如 model.py, 可以用 from .model import Autoencoder

# ==============================================================================
# 1. 从 sparse_autoencoder/train.py 复制的核心代码 (已恢复AuxK逻辑)
# ==============================================================================

# --- 并行通信简化 ---
@dataclass
class ShardingComms:
    def sh_allreduce_forward(self, x): return x
    def sh_allreduce_backward(self, x): return x
    def init_broadcast_(self, autoencoder): pass
    def dp_allreduce_(self, autoencoder): pass
    def sh_allreduce_scale(self, scaler): pass
    def sh_sum(self, x): return x
    def all_broadcast(self, x): return x

TRIVIAL_COMMS = ShardingComms()

# --- Autoencoder 模型 (恢复了AuxK逻辑) ---
class FastAutoencoder(nn.Module):
    def __init__(self, n_dirs_local, d_model, k, auxk, dead_steps_threshold, comms):
        super().__init__()
        self.n_dirs_local = n_dirs_local
        self.d_model = d_model
        self.k = k
        self.auxk = auxk
        self.comms = comms
        self.dead_steps_threshold = dead_steps_threshold
        
        self.encoder = nn.Linear(d_model, n_dirs_local, bias=False)
        self.decoder = nn.Linear(n_dirs_local, d_model, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(d_model))
        self.latent_bias = nn.Parameter(torch.zeros(n_dirs_local))
        
        self.register_buffer("stats_last_nonzero", torch.zeros(n_dirs_local, dtype=torch.long))
        
        self.decoder.weight.data = self.encoder.weight.data.T.clone()
        unit_norm_decoder_(self)

    def auxk_mask_fn(self, x):
        # 模拟原始代码中的dead_mask逻辑
        dead_mask = (self.stats_last_nonzero > self.dead_steps_threshold).float()
        x = x * dead_mask # 乘以0或1
        return x

    def forward(self, x):
        x_centered = x - self.pre_bias
        latents_pre_act = nn.functional.linear(x_centered, self.encoder.weight, self.latent_bias)
        
        # TopK for main loss
        vals, inds = torch.topk(latents_pre_act, self.k, dim=-1)
        
        # 更新神经元活跃度统计
        tmp = torch.zeros_like(self.stats_last_nonzero)
        tmp.scatter_add_(0, inds.reshape(-1), (vals > 1e-3).to(tmp.dtype).reshape(-1))
        self.stats_last_nonzero *= (1 - tmp.clamp(max=1))
        self.stats_last_nonzero += 1
        
        # TopK for AuxK loss
        auxk_vals, auxk_inds = None, None
        if self.auxk is not None:
            masked_latents = self.auxk_mask_fn(latents_pre_act.clone()) # clone to avoid in-place modification issues
            auxk_vals, auxk_inds = torch.topk(masked_latents, self.auxk, dim=-1)

        latents = torch.relu(vals)
        recons = self.decode_sparse(inds, latents)
        
        info = {
            "auxk_inds": auxk_inds,
            "auxk_vals": torch.relu(auxk_vals) if auxk_vals is not None else None,
        }
        
        return recons + self.pre_bias, info

    def decode_sparse(self, inds, vals):
        recons = torch.zeros(inds.shape[0], self.n_dirs_local, device=inds.device, dtype=vals.dtype)
        recons.scatter_(1, inds, vals)
        return self.decoder(recons)

def unit_norm_decoder_(autoencoder):
    autoencoder.decoder.weight.data /= autoencoder.decoder.weight.data.norm(dim=0, keepdim=True)

def unit_norm_decoder_grad_adjustment_(autoencoder):
    if autoencoder.decoder.weight.grad is None: return
    grad = autoencoder.decoder.weight.grad
    proj = torch.einsum("ij,ij->j", grad, autoencoder.decoder.weight.data)
    autoencoder.decoder.weight.grad -= proj * autoencoder.decoder.weight.data

class Logger:
    def __init__(self, **kws):
        self.vals = {}
        self.enabled = not kws.pop("dummy", False)
        if self.enabled:
            wandb.init(**kws)

    def logkv(self, k, v):
        if self.enabled:
            self.vals[k] = v.detach().item() if isinstance(v, torch.Tensor) else v
        return v

    def dumpkvs(self):
        if self.enabled:
            wandb.log(self.vals)
            self.vals = {}

def training_loop_(ae, train_acts_iter, loss_fn, lr, comms, eps, clip_grad, logger):
    opt = torch.optim.Adam(ae.parameters(), lr=lr, eps=eps, fused=True)
    
    for i, flat_acts_train_batch in enumerate(tqdm(train_acts_iter, desc="Training")):
        flat_acts_train_batch = flat_acts_train_batch.cuda()
        
        recons, info = ae(flat_acts_train_batch)
        loss = loss_fn(ae, flat_acts_train_batch, recons, info, logger)
        
        loss.backward()
        unit_norm_decoder_grad_adjustment_(ae)
        
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(ae.parameters(), clip_grad)
            
        opt.step()
        opt.zero_grad()
        unit_norm_decoder_(ae)
        
        logger.dumpkvs()

def batch_tensors(it: Iterable[torch.Tensor], batch_size: int) -> Iterator[torch.Tensor]:
    buffer = []
    current_size = 0
    for t in it:
        buffer.append(t)
        current_size += t.shape[0]
        while current_size >= batch_size:
            concatenated = torch.cat(buffer, dim=0)
            yield concatenated[:batch_size]
            buffer = [concatenated[batch_size:]]
            current_size -= batch_size
    if buffer and buffer[0].shape[0] > 0:
        yield torch.cat(buffer, dim=0)

def init_from_data_(ae, stats_acts_sample):
    # 检查 geom_median 是否已安装
    try:
        from geom_median.torch import compute_geometric_median
        median = compute_geometric_median(stats_acts_sample.float().cpu()).median.cuda().float()
        ae.pre_bias.data = median
    except ImportError:
        print("WARNING: 'geom_median' not found. Initializing pre_bias with mean instead.")
        median = stats_acts_sample.float().mean(dim=0).cuda()
        ae.pre_bias.data = median


# ==============================================================================
# 2. 我们自己实现的、针对HDF5的数据加载逻辑
# ==============================================================================
def create_h5_act_iterator(h5_path: str, d_model: int) -> Iterator[torch.Tensor]:
    print(f"INFO: 开始从 {h5_path} 流式加载激活值...")
    with h5py.File(h5_path, 'r') as f:
        dset = f['hidden_states']
        num_sequences = dset.shape[0]
        
        for i in range(num_sequences):
            seq_block = torch.from_numpy(dset[i, :, :])
            yield seq_block.reshape(-1, d_model)

def get_stats_sample(h5_path: str, num_samples: int) -> torch.Tensor:
    print(f"INFO: 从 {h5_path} 提取 {num_samples} 个样本用于初始化...")
    samples = []
    num_collected = 0
    with h5py.File(h5_path, 'r') as f:
        dset = f['hidden_states']
        for i in range(dset.shape[0]):
            seq_block = torch.from_numpy(dset[i, :, :])
            samples.append(seq_block.reshape(-1, dset.shape[2]))
            num_collected += samples[-1].shape[0]
            if num_collected >= num_samples:
                break
    return torch.cat(samples, dim=0)[:num_samples]

# ==============================================================================
# 3. 主程序：配置并启动训练
# ==============================================================================
@dataclass
class Config:
    # 与原始train.py保持一致
    h5_path: str = "/data0/yfliu/vqhlm/datasets/wikitext103_gpt2finetuned/train.h5"
    d_model: int = 768
    n_dirs: int = 32768
    bs: int = 4096 # token-level batch size
    k: int = 32
    auxk: int = 256
    auxk_coef: float = 1 / 32
    lr: float = 1e-4
    eps: float = 1e-8
    clip_grad: float | None = 1.0
    dead_toks_threshold: int = 10_000_000
    wandb_project: str = "sparse-autoencoder-full-logic"
    wandb_name: str | None = "gpt2-h5-resid-post-exp1-full"
    # 新增：保存模型的目录
    save_dir: str = "saved_models"

def normalized_mse(y_hat, y):
    return (y_hat - y).pow(2).sum() / y.pow(2).sum()

def main():
    cfg = Config()
    comms = TRIVIAL_COMMS

    acts_iter = create_h5_act_iterator(cfg.h5_path, cfg.d_model)
    stats_acts_sample = get_stats_sample(cfg.h5_path, num_samples=65536).cuda()

    ae = FastAutoencoder(
        n_dirs_local=cfg.n_dirs,
        d_model=cfg.d_model,
        k=cfg.k,
        auxk=cfg.auxk,
        dead_steps_threshold=cfg.dead_toks_threshold // cfg.bs,
        comms=comms,
    ).cuda()
    
    init_from_data_(ae, stats_acts_sample)

    mse_scale = 1 / (stats_acts_sample.var(dim=0).mean()).item()
    logger = Logger(project=cfg.wandb_project, name=cfg.wandb_name, config=cfg.__dict__)

    # 恢复了AuxK的损失函数
    def loss_fn(ae, flat_acts_train_batch, recons, info, logger):
        # 主MSE损失
        main_mse = (recons - flat_acts_train_batch).pow(2).mean()
        logger.logkv("train_mse_unscaled", main_mse)
        
        # AuxK损失
        auxk_loss = torch.tensor(0.0, device=main_mse.device)
        if info["auxk_inds"] is not None:
            auxk_recons = ae.decode_sparse(info["auxk_inds"], info["auxk_vals"])
            residual = (flat_acts_train_batch - recons).detach()
            auxk_loss = normalized_mse(auxk_recons, residual)
        
        logger.logkv("train_auxk_loss", auxk_loss)
        
        return main_mse * mse_scale + cfg.auxk_coef * auxk_loss

    training_loop_(
        ae,
        batch_tensors(acts_iter, cfg.bs),
        loss_fn,
        lr=cfg.lr,
        eps=cfg.eps,
        clip_grad=cfg.clip_grad,
        logger=logger,
        comms=comms,
    )

    ### =======================================================================
    ### 新增：训练完成后保存模型的逻辑
    ### =======================================================================
    print("\n" + "="*50)
    print("训练完成，正在保存模型...")

    # 1. 创建保存目录 (如果不存在)
    os.makedirs(cfg.save_dir, exist_ok=True)

    # 2. 确定文件名，如果 wandb_name 未设置，则使用默认名称
    model_name = cfg.wandb_name if cfg.wandb_name else "sae_model"
    save_path = os.path.join(cfg.save_dir, f"{model_name}.pt")

    # 3. 保存模型的 state_dict
    torch.save(ae.state_dict(), save_path)
    
    print(f"模型已成功保存到: {save_path}")
    print("="*50)


if __name__ == "__main__":
    main()
```

接下来是代码的解析：


### 计算图 (Mermaid)

![Alt text](<mermaid-202577 111231.png>)

### 数学公式详解

#### 前向传播公式

1. **② 中心化**：

   $$
   \text{E1: } \mathbf{x}_{\text{centered}} = \mathbf{x} - \mathbf{b}_{\text{pre}}
   $$

2. **③ 编码**：

   $$
   \text{E2: } \mathbf{z}_{\text{pre}} = \mathbf{W}_{\text{enc}} \mathbf{x}_{\text{centered}} + \mathbf{b}_{\text{lat}}
   $$

3. **④ Top-K 稀疏化**：

   $$
   \text{E3: } \mathbf{z} = \text{TopK}_k(\operatorname{ReLU}(\mathbf{z}_{\text{pre}}))
   $$

4. **⑤ 解码**：

   $$
   \text{E4: } \mathbf{x}_{\text{partial}} = \mathbf{W}_{\text{dec}} \mathbf{z}
   $$

5. **⑥ 添加偏置**：

   $$
   \text{E5: } \hat{\mathbf{x}} = \mathbf{x}_{\text{partial}} + \mathbf{b}_{\text{pre}}
   $$

6. **⑦ 主损失**：

   $$
   \text{E6: } \mathcal{L}_{\text{MSE}} = \frac{1}{d} \|\hat{\mathbf{x}} - \mathbf{x}\|^2_2
   $$

7. **⑧ 缩放**：

   $$
   \text{E7: } \mathcal{L}_{\text{scaled}} = \alpha \cdot \mathcal{L}_{\text{MSE}} \quad (\alpha = \text{mse\_scale})
   $$

8. **⑨ AuxK 稀疏化**：

   $$
   \text{E8: } \mathbf{z}_{\text{auxk}} = \text{TopK}_{\text{auxk}}(\operatorname{ReLU}(\mathbf{z}_{\text{pre}} \odot \mathbb{I}_{\text{dead}}))
   $$

9. **⑩ 残差计算**：

   $$
   \text{E9: } \mathbf{r} = (\mathbf{x} - \hat{\mathbf{x}}).\text{detach()}
   $$

10. **⑪ 辅助解码**：

    $$
    \text{E10: } \hat{\mathbf{r}} = \mathbf{W}_{\text{dec}}[:, \mathcal{I}] \mathbf{v} \quad (\mathcal{I} = \text{auxk\_indices})
    $$

11. **⑫ 辅助损失**：

    $$
    \text{E11: } \mathcal{L}_{\text{AuxK}} = \beta \cdot \frac{\|\hat{\mathbf{r}} - \mathbf{r}\|^2_2}{\|\mathbf{r}\|^2_2 + \epsilon} \quad (\beta = \text{auxk\_coef})
    $$

12. **⑬ 总损失**：

    $$
    \text{E12: } \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{scaled}} + \mathcal{L}_{\text{AuxK}}
    $$

#### 反向传播公式

1. **⑭ 反向传播起点**：

   $$
   \nabla_{\mathcal{L}_{\text{total}}} = 1
   $$

2. **⑬ 总损失梯度**：

   $$
   \nabla_{\mathcal{L}_{\text{scaled}}} = 1, \quad \nabla_{\mathcal{L}_{\text{AuxK}}} = 1
   $$

3. **⑧ 缩放梯度**：

   $$
   \nabla_{\mathcal{L}_{\text{MSE}}} = \alpha
   $$

4. **⑦ 主损失梯度**：

   $$
   \nabla_{\hat{\mathbf{x}}} = \frac{2\alpha}{d} (\hat{\mathbf{x}} - \mathbf{x})
   $$

5. **⑥ 添加偏置梯度**：

   $$
   \nabla_{\mathbf{x}_{\text{partial}}} = \nabla_{\hat{\mathbf{x}}}, \quad \nabla_{\mathbf{b}_{\text{pre}}} = \sum \nabla_{\hat{\mathbf{x}}}
   $$

6. **⑤ 解码梯度**：

   $$
   \nabla_{\mathbf{W}_{\text{dec}}^{\text{(main)}}} = \nabla_{\mathbf{x}_{\text{partial}}} \mathbf{z}^\top, \quad \nabla_{\mathbf{z}} = \mathbf{W}_{\text{dec}}^\top \nabla_{\mathbf{x}_{\text{partial}}}
   $$

7. **⑫ 辅助损失梯度**：

   $$
   \nabla_{\hat{\mathbf{r}}} = \frac{2\beta}{\|\mathbf{r}\|^2_2 + \epsilon} (\hat{\mathbf{r}} - \mathbf{r})
   $$

8. **⑪ 辅助解码梯度**：

   $$
   \nabla_{\mathbf{W}_{\text{dec}}^{\text{(aux)}}}[:, j] = 
   \begin{cases} 
   \nabla_{\hat{\mathbf{r}}_i} v_j & \text{if } j \in \mathcal{I} \\
   0 & \text{otherwise}
   \end{cases}
   $$

9. **④ Top-K 梯度**：

   $$
   \nabla_{\mathbf{z}_{\text{pre}, i}} = 
   \begin{cases} 
   \nabla_{\mathbf{z}_i} \cdot \mathbb{I}(z_{\text{pre},i} > 0) & \text{if } i \in S_k \\
   0 & \text{otherwise}
   \end{cases}
   $$

10. **③ 编码梯度**：

    $$
    \nabla_{\mathbf{W}_{\text{enc}}} = \nabla_{\mathbf{z}_{\text{pre}}} \mathbf{x}_{\text{centered}}^\top, \quad 
    \nabla_{\mathbf{b}_{\text{lat}}} = \sum \nabla_{\mathbf{z}_{\text{pre}}}
    $$

11. **② 中心化梯度**：

    $$
    \nabla_{\mathbf{x}_{\text{centered}}} = \mathbf{W}_{\text{enc}}^\top \nabla_{\mathbf{z}_{\text{pre}}}
    $$

12. **⑮ 梯度调整**：

    $$
    \nabla_{\mathbf{W}_{\text{dec}}}^{\text{(adj)}} = \nabla_{\mathbf{W}_{\text{dec}}} - \text{diag}(\mathbf{w}_j^\top \nabla_{\mathbf{W}_{\text{dec}}}) \mathbf{W}_{\text{dec}}
    $$

13. **⑰ 权重归一化**：

    $$
    \mathbf{W}_{\text{dec}}[:, j] \leftarrow \frac{\mathbf{W}_{\text{dec}}[:, j]}{\|\mathbf{W}_{\text{dec}}[:, j]\|_2}
    $$

### 梯度传播路径特性

1. **主路径梯度流**：
   ```
   ⑭ → ⑬ → ⑧ → ⑦ → ⑥ → ⑤ → ④ → ③ → ②
   ```
   更新参数：`W_enc, b_lat, b_pre, W_dec`（主激活列）

2. **辅助路径梯度流**：
   ```
   ⑭ → ⑬ → ⑫ → ⑪ → ⑨ → ③ → ②
   ```
   更新参数：`W_enc, b_lat, b_pre, W_dec`（辅助激活列）

3. **梯度切断点**：
   - 残差计算 `r = (x - x_hat).detach()` 阻止辅助损失梯度影响主重构路径
   - 确保辅助训练只激活"死亡神经元"，不干扰主特征学习

4. **参数更新范围**：

   | 参数        | 主路径更新 | 辅助路径更新 |
   |------------|-----------|------------|
   | `W_enc`    | ✓         | ✓          |
   | `b_lat`    | ✓         | ✓          |
   | `b_pre`    | ✓         | ✓          |
   | `W_dec`    | TopK列    | AuxK列     |

5. **梯度调整机制**：
   ```python
   # 正交投影伪代码
   for j in range(n_dirs):
       w_j = W_dec[:, j]
       g_j = grad_W_dec[:, j]
       # 投影到与w_j正交的方向
       grad_W_dec[:, j] = g_j - torch.dot(g_j, w_j) * w_j
   ```
