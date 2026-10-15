"""语料 memmap -> [BOS] doc [EOS] 流 -> 定长块；批次顺序可存进 checkpoint 恢复。

MASK id 不允许出现在语料里（数据自检）。恢复批次顺序是 C4 的根。
参考实现：common/data-example.py
"""
