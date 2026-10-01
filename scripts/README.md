# scripts/

环境、数据、训练和可视化相关的命令行脚本。

## 下载数据

先确保软链接已经建立：

```bash
./scripts/setup_symlinks.sh
```

脚本只下载有上限的样本，默认最多 10,000 篇，不会自动下载完整数据集：

```bash
.venv/bin/python scripts/download_data.py tinystories --max-docs 10000
.venv/bin/python scripts/download_data.py openwebtext --max-docs 10000
.venv/bin/python scripts/download_data.py wikipedia-zh --max-docs 10000
```

`datasets` 4.2.0 在 Python 3.14 + `dill` 0.4.0 下会报
`Pickler._batch_setitems() takes 2 positional arguments but 3 were given`。
脚本会自动回退到 `--backend parquet`：`hf_hub_download` 拉 parquet 分片 +
`pyarrow` 读列，不碰 pin 的版本。`--backend {auto,datasets,parquet}` 可强制。

输出分别写入：

```text
data/raw/tinystories.jsonl
data/raw/openwebtext.jsonl
data/raw/wikipedia-zh.jsonl
```

`data/` 实际指向 `/mnt/extra/data/`。可用 `--max-chars` 进一步限制下载量，用 `--output` 指定文件名。

## 下载 minimind_dataset

```bash
.venv/bin/python scripts/download_minimind.py --list        # 只列文件和大小
.venv/bin/python scripts/download_minimind.py               # 全量 ~24.5 GiB
.venv/bin/python scripts/download_minimind.py --only dpo rlaif
```

从 HF `jingyaogong/minimind_dataset` 拉全部 `.jsonl` 到 `data/raw/minimind_dataset/`，
中断后重跑会从 `.incomplete` 续传。`data/raw/` 下已有的
`pretrain_t2t_mini.jsonl` / `sft_t2t_mini.jsonl` 走 symlink，不重复下载。

全量 22 GiB 实测约 63 min（~6.6 MB/s，hf-mirror）。长下载建议在主会话跑，
不要挂 subagent —— 子进程会被 provider 中断一起杀掉。

## 准备 token 数据

默认使用 6400-vocab tokenizer，输出格式与现有 minimind 数据一致：

```bash
.venv/bin/python scripts/prepare_data.py \
  data/raw/tinystories.jsonl \
  --output data/prepared/tinystories-6400 \
  --append-eos
```

生成：

```text
data/prepared/tinystories-6400/tokens.bin
data/prepared/tinystories-6400/doc_lengths.npy
data/prepared/tinystories-6400/meta.json
```

Qwen3 数据必须传入 Qwen tokenizer，并单独放在自己的输出目录；不能与 6400-vocab token 混用：

```bash
.venv/bin/python scripts/prepare_data.py \
  data/raw/<qwen-jsonl> \
  --tokenizer /path/to/qwen/tokenizer.json \
  --output data/prepared/<name>-qwen
```

## 大文件约定

数据和 checkpoint 不写入仓库本地磁盘；先运行 `setup_symlinks.sh`，再通过 `data/` 和 `checkpoints/` 访问 `/mnt/extra`。训练日志保留在 `logs/run/`，不需要软链接。

其他脚本：

- `install_dllm.sh`：固定 commit、应用 patch 并 editable 安装 dllm；
- `setup_env.sh`：创建或更新项目 `.venv`；
- `plot.py`：绘制训练日志；
- `bench_throughput.py`：吞吐冒烟测试（只驱动 `transformers`，不实现模型）；
- `scripts/patches/`：保存 dllm 等第三方依赖所需的可复现补丁。

## 测吞吐

```bash
.venv/bin/python scripts/bench_throughput.py --seq-len 512 --batch 16
.venv/bin/python scripts/bench_throughput.py --sweep --steps 100 --warmup 25
```

报 `train-only` 和 `wall-clock` 两个 tok/s，方法学见 `notes/05-measurement.md`。

⚠️ **这张 880M 是核显，桌面和别的会话都在抢它。** 同一配置实测离散度可达
2.7 倍（8,824 / 6,054 / 3,251 tok/s）。测之前确认没有别的 GPU 任务，
同一配置跑 3 次报离散度，不要只跑一次。预热至少 50 步。
