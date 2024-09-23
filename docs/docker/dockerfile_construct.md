
# Lean4+Mathlib开发环境 Dockerfile 构建经验


## 1. Docker 命令概览

| 命令 | 描述 | 示例 |
|------|------|------|
| FROM | 指定基础镜像 | `FROM ubuntu:22.04` |
| ENV | 设置环境变量 | `ENV DEBIAN_FRONTEND=noninteractive` |
| RUN | 执行 shell 命令 | `RUN apt-get update && apt-get install -y ...` |
| USER | 切换当前用户 | `USER leanuser` |
| WORKDIR | 设置工作目录 | `WORKDIR /home/leanuser` |
| CMD | 设置容器启动时的默认命令 | `CMD [ "bash", "-l" ]` |

## 2. 最佳实践

1. 使用 `&&` 连接多个命令，减少 RUN 指令的数量，有助于减小镜像层数。
2. 清理不必要的文件（如 `apt-get clean`），减小镜像大小。
3. 使用非 root 用户运行应用，提高安全性。
4. 将频繁变动的命令放在 Dockerfile 的后面，利用 Docker 的缓存机制提高构建效率。

## 3. 详细解析

1. **基础镜像选择 (FROM)**
   ```dockerfile
   FROM ubuntu:22.04
   ```
   - 指定基础镜像，这里使用 Ubuntu 22.04 LTS 版本。

2. **环境变量设置 (ENV)**
   ```dockerfile
   ENV DEBIAN_FRONTEND=noninteractive
   ```
   - 设置环境变量，用于配置安装过程中的行为。

3. **包管理和系统依赖安装 (RUN)**
   ```dockerfile
   RUN apt-get update && apt-get install -y ...
   ```
   - 更新包列表并安装必要的系统依赖。

4. **用户创建和权限配置 (RUN, USER)**
   ```dockerfile
   RUN useradd -m -s /bin/bash -G sudo leanuser
   USER leanuser
   ```
   - 创建非 root 用户并配置 sudo 权限。
   - 切换到新创建的用户。

5. **工作目录设置 (WORKDIR)**
   ```dockerfile
   WORKDIR /home/leanuser
   ```
   - 设置工作目录，后续命令将在此目录下执行。

6. **Elan 安装 (RUN)**
   ```dockerfile
   RUN curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y
   ```
   - 使用官方脚本安装 Elan（Lean 版本管理器）。

7. **Lean 工具链安装 (RUN)**
   ```dockerfile
   RUN elan toolchain install $(cat lean-toolchain)
   ```
   - 安装指定版本的 Lean 工具链。

8. **项目初始化和构建 (RUN)**
   ```dockerfile
   RUN lake update
   RUN lake build
   ```
   - 使用 Lake 更新项目依赖并构建项目。

9. **环境变量配置 (RUN)**
   ```dockerfile
   RUN echo 'export LEAN_PATH="..."' >> ~/.bashrc
   ```
   - 配置 Lean 相关的环境变量。

10. **默认启动命令设置 (CMD)**
    ```dockerfile
    CMD [ "bash", "-l" ]
    ```
    - 设置容器启动时的默认命令。




## 4. 附录：原始Dockerfile
```dockerfile
# 使用官方的 Ubuntu 22.04 LTS 作为基础镜像
FROM ubuntu:22.04

# 设置环境变量以避免在安装过程中出现交互提示
ENV DEBIAN_FRONTEND=noninteractive

# 更新包列表并安装必要的系统依赖
RUN apt-get update && apt-get install -y \
    sudo \
    git \
    curl \
    bash-completion \
    python3 \
    python3-requests \
    build-essential \
    libffi-dev \
    libssl-dev \
    pkg-config \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 创建一个非 root 用户以增强安全性
RUN useradd -m -s /bin/bash -G sudo leanuser \
    && echo "leanuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# 切换到非 root 用户
USER leanuser
WORKDIR /home/leanuser

# 安装 Elan（Lean 版本管理器）
RUN curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y

# 将 Elan 的路径添加到环境变量中
ENV PATH="/home/leanuser/.elan/bin:${PATH}"

# 安装 Lean 工具链（根据 mathlib 的 lean-toolchain 文件）
RUN curl -s https://raw.githubusercontent.com/leanprover-community/mathlib4/master/lean-toolchain -o lean-toolchain \
    && elan toolchain install $(cat lean-toolchain) \
    && elan default $(cat lean-toolchain)

# 验证 Lake 安装
RUN lake --version

# 手动创建 Lean 项目并添加 Mathlib 作为依赖
RUN mkdir my_project && \
    cd my_project && \
    echo 'import Lake\nopen Lake DSL\n\npackage «my_project» where\n  -- add package configuration options here\n\nrequire mathlib from git\n  "https://github.com/leanprover-community/mathlib4.git"\n\n@[default_target]\nlean_lib «MyProject» where\n  -- add library configuration options here' > lakefile.lean

# 设置工作目录为项目目录
WORKDIR /home/leanuser/my_project

# 创建 MyProject.lean 文件
RUN echo 'def hello := "Hello from MyProject!"' > MyProject.lean

# 初始化项目并更新依赖
RUN lake update

# 构建项目
RUN lake build

# 创建测试文件
RUN echo 'import Mathlib\n\ndef main : IO Unit :=\n  IO.println s!"Hello from Mathlib! {2 + 2}"\n\n#eval main' > test_mathlib.lean

# 设置环境变量以包含所有必要的库路径
RUN echo 'export LEAN_PATH="$LEAN_PATH:$(find .lake/packages -name lib -type d | tr "\n" ":" | sed "s/:$//")"' >> ~/.bashrc

# 确保 .bashrc 在每次启动时都被加载
RUN echo 'source ~/.bashrc' >> ~/.profile

# 设置默认的启动命令
CMD [ "bash", "-l" ]

```

## 5. 原始Dockerfile详解

## 基础镜像选择

```markdown
FROM ubuntu:22.04
```

- 选择官方 Ubuntu 22.04 LTS 作为基础镜像，确保稳定性和长期支持。
- 考虑使用更轻量的基础镜像（如 Alpine）可能会带来兼容性问题，因此选择 Ubuntu 是安全的选择。

## 系统依赖安装

```markdown
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    sudo git curl bash-completion python3 python3-requests \
    build-essential libffi-dev libssl-dev pkg-config \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
```

- 设置 `DEBIAN_FRONTEND=noninteractive` 避免安装过程中的交互提示。
- 安装必要的系统依赖，包括编译工具和 Python 环境。
- 使用 `apt-get clean` 和删除 `/var/lib/apt/lists/*` 减小镜像大小。

## 用户安全性

```markdown
RUN useradd -m -s /bin/bash -G sudo leanuser \
    && echo "leanuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER leanuser
WORKDIR /home/leanuser
```

- 创建非 root 用户 `leanuser` 增强安全性。
- 将用户添加到 sudo 组并配置无密码 sudo 权限，方便后续操作。
- 切换到非 root 用户，设置工作目录。

##  Elan 安装

```markdown
RUN curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y

ENV PATH="/home/leanuser/.elan/bin:${PATH}"
```

- 使用官方脚本安装 Elan（Lean 版本管理器）。
- 将 Elan 的路径添加到环境变量中，确保可以全局访问。

## Lean 工具链安装

```markdown
RUN curl -s https://raw.githubusercontent.com/leanprover-community/mathlib4/master/lean-toolchain -o lean-toolchain \
    && elan toolchain install $(cat lean-toolchain) \
    && elan default $(cat lean-toolchain)
```

- 从 mathlib 仓库获取最新的 `lean-toolchain` 文件。
- 安装指定版本的 Lean 工具链并设置为默认版本。

## 项目初始化

```markdown
RUN mkdir my_project && cd my_project && \
    echo 'import Lake\nopen Lake DSL\n\npackage «my_project» where\n  -- add package configuration options here\n\nrequire mathlib from git\n  "https://github.com/leanprover-community/mathlib4.git"\n\n@[default_target]\nlean_lib «MyProject» where\n  -- add library configuration options here' > lakefile.lean

WORKDIR /home/leanuser/my_project

RUN echo 'def hello := "Hello from MyProject!"' > MyProject.lean

RUN lake update
RUN lake build
```

- 手动创建 Lean 项目结构，包括 `lakefile.lean` 和主文件。
- 使用 `lake update` 初始化项目并更新依赖。
- 使用 `lake build` 构建项目，确保环境正常。

## 环境变量配置

```markdown
RUN echo 'export LEAN_PATH="$LEAN_PATH:$(find .lake/packages -name lib -type d | tr "\n" ":" | sed "s/:$//")"' >> ~/.bashrc
RUN echo 'source ~/.bashrc' >> ~/.profile
```

- 配置 `LEAN_PATH` 环境变量，包含所有必要的库路径。
- 确保每次启动容器时都加载这些环境变量。

## 启动命令

```markdown
CMD [ "bash", "-l" ]
```

- 设置默认启动命令为登录 shell，确保环境变量被正确加载。

## 6. Lean4 Dockerfile 构建经验总结

1. **版本控制**：使用 `lean-toolchain` 文件确保 Lean 版本与 Mathlib 兼容。
2. **依赖管理**：通过 Elan 和 Lake 管理 Lean 和项目依赖，简化版本控制。
3. **安全性**：使用非 root 用户运行应用，提高容器安全性。
4. **环境变量**：正确配置 `LEAN_PATH` 确保所有库路径可访问。
5. **构建验证**：在 Dockerfile 中进行项目构建，验证环境配置正确。
6. **镜像优化**：清理不必要的文件，减小最终镜像大小。
