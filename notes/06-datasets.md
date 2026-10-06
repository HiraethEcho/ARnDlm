# 数据集 — 来源、格式与实测压缩比

## 1. minimind_dataset

来源: HF dataset [`jingyaogong/minimind_dataset`](https://huggingface.co/datasets/jingyaogong/minimind_dataset)
(国内走 `HF_ENDPOINT=https://hf-mirror.com`; README 里也给了 ModelScope 链接)。
文件大小取自 repo 的 `siblings[].size`, 由 `scripts/download_minimind.py --list` 实测输出。
行数为本地下载后 `wc -l` 实测。

| 文件 | 大小 | 行数 | 用途 | 推荐 `max_seq_len` |
|---|---:|---:|---|---:|
| `pretrain_t2t_mini.jsonl` | 1.16 GiB | — | 轻量预训练 | ~768 |
| `pretrain_t2t.jsonl` | 7.71 GiB | 8,468,827 | 主线预训练 | ~380 |
| `sft_t2t_mini.jsonl` | 1.62 GiB | — | 轻量 SFT (混了 Tool Call) | ~768 |
| `sft_t2t.jsonl` | 13.13 GiB | 5,109,432 | 主线 SFT | — |
| `dpo.jsonl` | 53.6 MB | 17,166 | DPO 偏好对 | — |
| `rlaif.jsonl` | 23.8 MB | 19,502 | PPO / GRPO / CISPO rollout | — |
| `agent_rl.jsonl` | 82.0 MB | — | 多轮 Tool-Use agentic RL | — |
| `agent_rl_math.jsonl` | 18.4 MB | — | RLVR 数学补充 | — |
| `lora_exam.jsonl` | 24.7 MB | — | LoRA 微调示例 | — |
| `lora_medical.jsonl` | 34.0 MB | — | LoRA 微调示例 (医疗) | — |
| `lora_identity.jsonl` | 22.8 KB | 91 | LoRA 微调示例 (自我认知) | — |

推荐组合: `pretrain_t2t` + `sft_t2t` + `rlaif`/`agent_rl` 阶段式。

## 2. 两种行格式 —— 决定能不能直接喂 `prepare_data.py`

| 阶段 | 一行长什么样 | `scripts/prepare_data.py` |
|---|---|---|
| Pretrain | `{"text": "..."}` | ✅ 直接可用 |
| SFT / RL / DPO | `{"conversations": [{"role": "user", "content": ...}, ...]}` | ❌ 产出 0 token |

`read_text()` 只取 `row["text"]`。对话数据缺这个键, 会被 `if isinstance(text, str)`
静默跳过 —— **不报错, 只是 token 数为 0**, 很容易误判成「语料是空的」。
走对话数据必须先套 chat template 展平成一条 `text` (见 `notes/01-pretrain-sft-rl.md`)。

assistant 消息还带 `reasoning_content` 字段 (思维链), 展平时要不要保留是独立决定。

## 3. 6400-vocab BPE 的实测压缩比

`tokenizer/tokenizer.json` (6400 vocab, 6108 merges) 在本项目数据上的实测
(2026-10-01, 字符数由 raw jsonl 现算, token 数由 `meta.json`):

| 数据 | tok | 字符 | chars/token |
|---|---:|---:|---:|
| `minimind3-6400` (`pretrain_t2t_mini`) | 332,495,324 | 486,491,829 | **1.46** |
| `tinystories-6400` | 3,110,358 | 8,674,761 | 2.79 |
| `openwebtext-6400` | 17,174,450 | 48,594,444 | 2.83 |
| `wikipedia-zh-6400` | 38,571,096 | 36,123,690 | 0.94 |
| `wikitext-6400` (2026-10-05 加) | 194,509,960 | 535,934,762 | 2.75 |

minimind 官方 tokenizer 在中文上是 1.5~1.7 chars/token, 本项目 6400 BPE 实测 1.46
—— 同量级。**6400 词表对中文是超标的**: 中文 `wikipedia-zh` 达到 1.07 token/char,
即比字符数还多, 词表根本没压住中文。中文语料 + 6400 词表 = 每 token 成本按字符付。

选 `max_seq_len` 时按 chars/token 换算, 不要按字符数拍。

## 4. 本项目实际用哪份

`SPEC.md` 3.3 定的 6400-vocab track 用 `prepared/minimind3-6400/`
(= `pretrain_t2t_mini`, 332M token)。AR / MiniMind-A2D / dllm-A2D / Direct-DLM
四条路线共享这一份, 保证对照公平。其余数据集是横向参照, 不进主实验。