如何将文件从容器内移动到宿主机上
```cmd
docker cp <容器ID>:<容器内文件路径> <宿主机目标路径>
```
```cmd
docker cp <容器名称>:<容器内文件路径> <宿主机目标路径>
```

如何从宿主机上把文件传输到本机
```cmd
scp -p 1233 inuyasha@111.186.37.25:/home/inuyasha/data/bayuan_for_compile.json /Users/Inuyasha/Coding/Lean4Syntho/Semantic_Check/1101_Dataset_to_DPO
```
注意，Windows路径应该使用正斜杠，同时不需要加系统盘开头