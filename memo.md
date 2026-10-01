This repo is for
学习python，pytorch, transformers, dlm
复现项目，实验 DLM 并和 AR 比较

用相同的 uv 下 venv，相同的 tokenizer，vocab 等参数和相同的数据集
期间用工具监视训练情况，记录loss曲线等

- 首先训练基准模型：
  - 用 transformers 包预训练一个模型，用 lora, peft 包等后训练
  - 参考 minimind ，自己写训练循环等
- 然后做 A2D，分别基于前面训练的基础模型和开源社区的模型
  - 参考 minimind-dlm 手动实现
  - 用 dlm 包实现
- 最后基于 dllm 包直接训练一个 DLM 模型

references
~/Projects/dlm ~/Projects/minimind
~/Public/dlm/ (this is dlm package) ~/Public/minimind/ ~/Public/ChatLM-mini-Chinese/
minimind-dlm 的来源 https://jingyaogong.github.io/space/dllm_intro

当前任务，比较 SPEC 和 SPEC-gpt ， 重新设计 scaffold。
src/ 按照 SPEC-gpt 的风格，直接放在根目录下，不用 src/ 文件夹
使用 SPEC-gpt 版的命名方式

快速验证 pipeline 用更小的模型参数，仅仅为了验证 pipeline 是否可行，训练是否能跑通，loss 是否下降
```yaml
hidden_size: 256
num_layers: 2
num_attention_heads: 2
num_key_value_heads: 1
intermediate_size: 768
context_length: 128
```

先做下列任务，每完成一项commit一次
1. 先确定一版 SPEC
2. update scaffold
3. 处理 uv 环境，适配这些目标

