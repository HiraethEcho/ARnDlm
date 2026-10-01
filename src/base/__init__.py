"""base — 训练 / 后训练的公用工具。所有路线的底座。

放这里的东西必须**不依赖具体路线**:
    tokenizer.json      minimind BPE, vocab 6400, 6108 merges
                        PAD 0 <|endoftext|>  BOS 1 <|im_start|>
                        EOS 2 <|im_end|>     MASK 27 <|buffer1|>
    dataset.py          读 tokens.bin (uint16) + doc_lengths.npy
    packing.py          文档拼接成定长块 (本机吞吐高 1.71x)
    masking.py          MDM 加噪, p_mask = 1 - alpha(t)
    objective.py        AR 的 next-token CE 和 DLM 的加权 CE
    config.py           configs/*.yaml 的加载 (base -> 实验, 后者覆盖)

**objective 里 AR 和 DLM 的 loss 必须分开写, 不要合并。** 两条路径要能
在同一个 checkpoint 上对拍, 混在一起就没法判断是谁错了。

依赖: torch, tokenizers, numpy, pyyaml
"""
