"""下载/整理语料为 tokens.bin + doc_lengths.npy + meta.json（提前准备，与训练无关）。

输入是 minimind 数据集；conversations 行格式不能直接喂（SKIP 已知坑）。
参考实现：common/prepare-example.py
"""
