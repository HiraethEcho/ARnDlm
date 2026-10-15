"""四门槛：assistant eval loss 比、rep-4 < 0.5、三 prompt 格式、不回声；产出报告。

正式评测几百 token x >=5 prompt（rep-4 只在短样本上会跳）。
参考实现：src/eval/chat-example.py
"""
