# scripts/

环境、数据、训练和可视化相关的命令行脚本。

数据相关脚本规划：

- `download_data.py`：下载原始语料或公开数据集，写入软链接目录 `data/raw/`（实际位置为 `/mnt/extra/data/raw/`）；
- `prepare_data.py`：使用指定 tokenizer 编码、切分文档并生成软链接目录 `data/prepared/<name>/`（实际位置为 `/mnt/extra/data/prepared/<name>/`）；
- `setup_symlinks.sh`：把仓库内的 `data/` 和 `checkpoints/` 连接到 `/mnt/extra/`。

`data/` 目录只放数据读取、packing、masking 和 Dataset 代码；下载和生成数据的命令入口不放在 `data/`。

## 大文件约定

数据和 checkpoint 不写入仓库本地磁盘；先运行 `setup_symlinks.sh`，再通过 `data/` 和 `checkpoints/` 访问 `/mnt/extra`。训练日志保留在 `logs/run/`，不需要软链接。

- `scripts/patches/`：保存 dllm 等第三方依赖所需的可复现补丁。
