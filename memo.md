This repo is for
学习python，pytorch, transformers, dlm
复现项目，实验 DLM 并和 AR 比较

用相同的 uv 下 venv，相同的 tokenizer，vocab 等参数和相同的数据集
首先训练基准模型：

- 用 transformers 包预训练一个模型，用 lora, peft 包等后训练
- 参考 minimind ，自己写训练循环等

然后做 A2D，分别基于前面训练的基础模型和开源社区的模型

- 参考 minimind-dlm 手动实现
- 用 dlm 包实现

最后基于 dlm 包直接训练一个 DLM 模型

期间用工具监视训练情况，记录loss曲线等

第一项任务是参考 ~/Projects/dlm 实现 scaffold，做下列调整：logs/dev/ 放开发日志， logs/run/ 放运行日志；test/ example/ 的注释更精简。
scaffold 完成后做第一次 commit

commit 前不要执行训练相关的高占用程序

references
~/Projects/dlm ~/Projects/minimind
~/Public/dlm/ (this is dlm package) ~/Public/minimind/ ~/Public/ChatLM-mini-Chinese/
