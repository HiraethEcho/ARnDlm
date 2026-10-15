"""用 transformers 构 Qwen3：ModelCfg -> Qwen3Config -> Qwen3ForCausalLM。

字段全部显式写（不依赖库默认值）；种子在构造之前打。
参考实现：src/models/trans-example.py
"""
