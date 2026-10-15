"""SFT prepared 目录的仓内生产者（与 prepare/prepare.py 并列，负责它的写者一半）。

输入 minimind sft_t2t_*.jsonl（conversations 格式），输出 tokens.bin（预切块）+ assistant_spans.npy
（[n_spans,3] = block,start,end）+ meta.json。模板 user\\n{q} [EOS] assistant\\n{a} [EOS]，span 含轮末 EOS。
参考实现：common/sft-prepare-example.py
"""
