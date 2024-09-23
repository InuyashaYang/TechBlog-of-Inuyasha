

# Docker镜像部署与发布

| 命令 | 用途 |
|------|------|
| `docker build -t <image_name>:<tag> .` | 构建Docker镜像 |
| `docker run -it --rm <image_name>:<tag>` | 运行Docker容器 |
| `docker tag <image_name>:<tag> <username>/<image_name>:<tag>` | 为镜像添加新标签 |
| `docker push <username>/<image_name>:<tag>` | 推送镜像到仓库 |

## 1. 镜像构建

```bash
docker build -t lean4-mathlib:latest .
```

参数解释：
- `-t lean4-mathlib:latest`: 为构建的镜像指定名称和标签
- `.`: 指定Dockerfile所在的当前目录作为构建上下文

- 使用当前目录的Dockerfile构建镜像
- 标记镜像为`lean4-mathlib:latest`

## 2. 本地测试

```bash
docker run -it --rm lean4-mathlib:latest
```

参数解释：
- `-it`: 以交互模式运行容器，并分配一个伪终端
- `--rm`: 容器停止运行后自动删除
- `lean4-mathlib:latest`: 指定要运行的镜像名称和标签

- 交互式运行容器
- `--rm`选项确保容器停止后自动删除

## 3. 镜像推送（可选）

```bash
docker tag lean4-mathlib:latest username/lean4-mathlib:latest
docker push username/lean4-mathlib:latest
```

参数解释：
- `docker tag`: 为镜像创建一个新的标签
- `lean4-mathlib:latest`: 源镜像名称和标签
- `username/lean4-mathlib:latest`: 目标镜像名称和标签，通常包含Docker Hub用户名
- `docker push`: 将镜像推送到远程仓库

- 重新标记镜像以匹配Docker Hub仓库名
- 推送镜像到Docker Hub

