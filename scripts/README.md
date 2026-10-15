# scripts/ — 执行入口

| 文件 | 干什么 |
|---|---|
| `setup_env.sh` | uv 建 `.venv`（Python 3.14 + ROCm 10.0 wheel） |
| `install_dllm.sh` | clone + 打补丁（`patches/`）+ `--no-deps` 装 dllm |
| `setup_symlinks.sh` | `data/` `checkpoints/` 软链到 `/mnt/extra` |
| `chat.py` | 四门槛评测（骨架；参考实现 `src/eval/chat-example.py`） |
| `repl.py` | 终端 REPL（骨架；参考实现 `src/eval/repl-example.py`） |
| `collect_results.py` | 汇总 `summary.json` → `RESULTS-smoke.md`；`--gate` 当硬门禁 |
| `make_wikitext_jsonl.py` | wikitext-103 parquet → jsonl |
