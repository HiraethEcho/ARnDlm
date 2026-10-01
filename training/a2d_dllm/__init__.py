"""dllm-a2d — 用 dllm 包的工具做 A2D。

对应 PLAN.md 的 T2b。跟 a2d/ 做同一件事, 只是不手写。

    convert.py           调 dllm/pipelines/a2d/convert.py
    train.py             调 dllm 的 Trainer (MDLM objective)
    sample.py            调 dllm 的 sampler

**dllm 的 A2D 只实现了三个架构**: llama / qwen2 / qwen3
    ~/Public/dllm/dllm/pipelines/a2d/models/
所以基准模型必须是这三者之一。本项目选 Qwen3。

dllm/pipelines/a2d/models/qwen3/modeling_qwen3.py 是**拷贝 transformers 4.57
的 modeling_qwen3.py 改出来的** —— 想知道 A2D 到底改了什么, 直接对着
读这两份文件的 diff, 比读文档快。

依赖: dllm (--no-deps 装的), ../base
"""
