"""src/eval/repl.py -- 终端 REPL：交互查看模型输出。

两种模式：pretrain（续写，encode(bos=True) 直接续）与 chat（对话，模板
[BOS] user\\n{q} [EOS] assistant\\n 逐轮累积，EOS 停）。ckpt 目录自动识别
HF（有 config.json）与 native（只有 model.safetensors）两种存法。
参考实现：src/eval/repl-example.py
"""
