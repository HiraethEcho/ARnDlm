# configs/practical/

实践版配置。**空是有意的** —— 规模要等吞吐阶梯测完再定
（`example/teach/13-tier1.md`：先测参数量 × seq_len × batch 的阶梯，再拍数字）。

写第一份配置时照抄 `configs/example/tier0.yaml` 的结构：同样的字段、同样的
`version:` / `tier:` 头。字段名必须一致，因为两版走同一个加载器
`example/common/version.py::load_config`，字段不同就等于两条代码路径。

标称候选（SPEC 4.2，**未拍板**）：

```yaml
model:
  hidden_size: 512
  num_layers: 8
  num_heads: 8
  num_kv_heads: 4
  intermediate_size: 1408
  context_length: 512
```

数据也要换：实验版用 tinystories-6400 切片，实践版用 minimind3-6400（332M token）。
