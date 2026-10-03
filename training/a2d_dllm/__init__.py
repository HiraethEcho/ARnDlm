"""dllm-a2d — 用 dllm 包的工具做 A2D。

对应教学计划 Step C4 (`example/teach/09-a2d-dllm.md`)。跟 ../a2d_minimind/ 做同一件事, 只是不手写。

    convert.py           调 dllm/pipelines/a2d/convert.py
    train.py             调 dllm 的 Trainer (MDLM objective)
    sample.py            调 dllm 的 sampler

**上游是 SFT 之后的权重, 不是预训练权重。** 学习路径是
`ar_hf` -> `sft_ar_hf` -> `a2d_dllm`, 所以这里 load 的是 SFT 的 checkpoint。
不先有能 chat 的 AR 权重, 就无法判断 DLM 的输出是「变差了」还是「本来就差」。

**dllm 的 A2D 只实现了三个架构**: llama / qwen2 / qwen3
    ~/Public/dllm/dllm/pipelines/a2d/models/
所以基准模型必须是这三者之一。本项目选 Qwen3。

dllm/pipelines/a2d/models/qwen3/modeling_qwen3.py 是**拷贝 transformers 4.57
的 modeling_qwen3.py 改出来的** —— 想知道 A2D 到底改了什么, 直接对着
读这两份文件的 diff, 比读文档快。**不要升到 transformers 5.x**, 那个文件的
`create_causal_mask` 是照 4.57 的签名写的。

**steps 参数有两个语义**: 本项目 `--dlm-steps` 是**每块**步数, dllm 的同名参数是
**全序列**预算(内部会 `ceil(steps / num_blocks)` 摊到每块)。传错不会报错,
只会让解码工作量差好几倍。见 example/ISSUES.md DONE-3。

依赖: dllm (--no-deps 装的), ../../diffusion, ../sft_ar_hf
"""
