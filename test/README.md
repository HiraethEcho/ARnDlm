# test/ — 验证脚本

每个脚本只回答一个问题。**不是**性能 benchmark —— 硬件数字在
`logs/dev/02-hardware-perf.md`，换机器重测，不进 git。

## 跑法

```sh
.venv/bin/python test/01-tokenizer-mask.py
```

不用切目录（没有 `from _model import` 那种同目录 import）。

## 索引

| 脚本 | 回答什么 | 需要 |
|---|---|---|
| `01-tokenizer-mask.py` | vocab 6400 对不对？mask token id 27 真的空闲吗？tokenizer 够快吗？ | — |
| `02-data-pipeline.py` | `tokens.bin` 读对了吗？packing 拼接后有没有跨文档污染？ | `data/` 软链接 |
| `03-lora-inject.py` | `peft` 挂到哪层？参数量变化多少？哪些模块**没**被冻结？ | `peft` |
| `04-dllm-loss-check.py` | 手算的 MDLM 加权 loss 和 `dllm` 的实现对得上吗？ | `dllm` |
| `05-repetition-rate.py` | AR 和 DLM 的重复率差多少？（核心指标） | T1/T2 权重 |

03/04 跳过的条件是「对应包没装」，不是静默通过。

## 注释约定

这是学 Python 的项目，注释解释 **Python 用法**：
`nn.Module` / `nn.Parameter` / `super().__init__()` 的作用、
`view` vs `reshape` vs `transpose`、`torch.where` 的广播、
`try/except` 捕获 OOM、`del` + `empty_cache()` 为什么要手动写。

**但要精简** —— 一行只回答「这行为什么这么写」。
不写「`#` 是单行注释」这种废话。

## 新脚本的规矩

- 结尾必须有明确的 `PASS` / `FAIL`，不能跑完什么都不说。
- 打印**实测数字**，不打印「正常」「很好」。
- 失败时 `sys.exit(1)`，这样以后能进 CI。
