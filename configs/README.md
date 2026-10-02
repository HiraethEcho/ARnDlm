# configs/

配置按**版本**分目录（SPEC 1.1.1）。两个版本走同一个加载器：

```python
from common.version import load_config, cell_path
cfg = load_config("example", "tier0")        # configs/example/tier0.yaml
ckpt = cell_path("ar-3.2m-hf", "example")   # checkpoints/ar-3.2m-hf-example
```

| 目录 | 版本 | 加载 | 用途 |
|---|---|---|---|
| `example/` | 实验版 | `load_config("example", "tier0")` | pipeline 能不能跑通，Tier 0 / 3.2M |
| `practical/` | 实践版 | `load_config("practical", "tier1")` | 正式实验，Tier 1 / ~50M |

`-example` 后缀由 `cell_path()` 统一加，不在各调用点拼：拼错不报错，
只会静默覆盖另一个版本的权重。

`practical/` 还没有内容 —— 实践版的规模要等吞吐阶梯测完再定（`example/teach/13-tier1.md`）。
