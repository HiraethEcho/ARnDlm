"""utils/data.py -- 预 tokenize 的二进制语料 -> 定长打包块，加上一个能存进 checkpoint 的批次顺序。

用途
    读 `data/prepared/<corpus>/` 的三个文件（`tokens.bin` / `doc_lengths.npy` /
    `meta.json`），拼成 `[BOS] doc [EOS] [BOS] doc [EOS] ...` 的一条流，切成定长块，
    并按固定顺序喂给训练循环。

    对应 example/common/data.py（196 行）。**这是 A0 里唯一一处「写错了不报错」的层**：
    偏移表写错，出来的仍然是通顺文本，只是每篇文档错位几个 token；loss 照样降。
    所以本文件的验收不是「跑通了」，而是「逐条对上了手工构造的小语料」。

    Step 0 的 teach 文档把这一层的三个坑全列在 `example/teach/00-common-base.md` §0.2，
    本文件的「注意」就是那三条 + 实测踩到的两条。

要写什么

        @dataclass PackedData
            blocks: torch.Tensor      # [n_blocks, block_size] int32
            n_docs / n_tokens / block_size / prepared_dir: Path
            __len__() -> n_blocks
            batch(idx) -> blocks[idx].long()      # 转 int64 只在这里做，见注意 5

        ids_of(batch) -> Tensor
            `batch[0] if batch.dim() == 3 else batch`。GOAL.md 清单第 1 条就是它。

        _load(prepared_dir) -> (meta, tokens, lengths)
            memmap，`dtype="<u2"`。

        _stream(tokens, lengths) -> np.ndarray      # int32 一维
            双偏移表那一段，见注意 1。

        build_packed(prepared_dir, block_size, max_blocks) -> PackedData
            四道自洽检查（注意 2）+ 切块 + 丢尾巴。

        class BlockStream
            __init__(n_blocks, batch_size, seed)
            next() -> Tensor            # 满一个 epoch 就换 permutation
            state() / load_state(state) # 见注意 3

        fixed_batches(n_batches, batch_size, n_blocks, seed=1234) -> list[Tensor]

注意
    1. **两张偏移表，不能混。**
       `starts = cumsum(lengths + 2)` 是**流里**的位置（含 BOS/EOS 的两个空位），
       `doc_starts = cumsum(lengths)` 是**token 文件里**的位置（不含）。
       用错哪张，正文 token 全部整体偏移，而文本看起来仍然通顺 —— 实验版第一版就这么错，
       是 `example/tests/test_common.py` 的「手工构造三篇文档」把它抓住的。
       最容易写错的是那两行的**不对称**：`starts` 对 `lengths+2` 做全量 cumsum（写进
       `starts[1:]`，长度 n+1），`doc_starts` 只对 `lengths[:-1]` 做（写进 `doc_starts[1:]`，
       长度 n）。第二个少一项是因为第 i 篇的起点只取决于前 i-1 篇的长度；两个都写成全量
       就多一项，索引时错一位。

    2. **四道自洽检查，一道都不能省**（`build_packed` 开头）：
       a. `len(lengths) == meta["n_docs"]` 且 `lengths.sum() == tokens.shape[0]`
          —— 三件文件互相对账。
       b. `meta["vocab_size"] == config.VOCAB` —— 换过 BPE 的语料会索引错 embedding 行，
          唯一的症状是 loss 高得说不清。
       c. `stream.max() < VOCAB` —— meta 可以自洽而二进制里藏着一个越界 id。
       d. 语料里**不能出现 MASK 的 id**。A2D 靠「这个位置是 MASK」判断哪些位置要预测；
          语料里本来就有的话，它就和一个被噪声遮住的真实槽无法区分。
       这四条都便宜（一次 pass），漏掉任何一条的排查成本都是「训练几小时再回头」。

    3. `BlockStream` 存的是 `(epoch, pos, n_blocks, batch_size)`，permutation 由
       `(seed + epoch)` **重新生成**而不是推进。`load_state` 必须校验几何：用不同的
       `--batch-size` 或 `--max-blocks` 续训时，`pos` 的含义整个变了，而曲线看起来正常
       —— 这是 acceptance C4（`N 步 + resume` 逐位等于 `N+K 步` 直跑）成立的前提。
       为什么不用 `DataLoader`：它的 sampler 状态存不下来。

    4. **尾巴丢掉，不要 padding。** `n_blocks = len(stream) // block_size`。
       补 PAD(0) 的话 PAD 就进了训练目标，模型学着预测 PAD —— 除非你显式把 labels 置
       -100，而打包路径上根本不需要这个复杂度。

    5. 块存 int32，`batch()` 时才 `.long()`。索引张量要 int64，但 65 万块 × 512 全存
       int64 是 2.6 GB，白吃一倍内存。转换发生在 gather 之后，一次只有 `[B, L]` 个元素。

    6. `fixed_batches` 用**独立的 seed（默认 1234）**，和 BlockStream 的
       `seed` 分开。两者同源的话，评测批次就是训练批次的前几批，eval loss 会偏低
       —— 那不是模型好，是题目泄漏。本项目的验证集就是这个冻结批次集合（没有 held-out
       split），理由是跨实现比 loss 必须在**同一批 token** 上比，否则测的是采样器。

    7. `ids_of` 的 dim==3 分支现在没有调用方（SFT 才有 `[2, B, L]`）。要么现在写上并
       在 docstring 说明「为 B1 准备」，要么等到 B1 再加 —— 但别留一个没人调的分支
       还不写原因（GOAL.md 清单第 7 条）。这条由你判断。

依赖
    numpy（memmap / cumsum 是它的活）、torch（只做块张量和 randperm）、
    utils/config.py（VOCAB / MASK / BLOCK_SIZE / DATA_DIR）。不 import device.py，
    不 import torch.nn —— 这一层与硬件无关，才能在 CPU 上先对拍。

验收
    先跑不依赖本文件的那条，证明语料本身自洽：

        .venv/bin/python test/02-data-pipeline.py

    然后是**手工小语料**这一条（本文件真正的验收）。构造三篇文档、长度各不相同，
    手工算出流里每个 BOS/EOS 的下标，断言：

        .venv/bin/python -c "
        import numpy as np
        from utils.data import _stream
        lengths = np.array([3, 1, 4], dtype=np.int64)
        tokens  = np.arange(1, 9, dtype='<u2')      # 8 个正文 token
        s = _stream(tokens, lengths).tolist()
        print(s)
        assert s == [1,1,2,3,2, 1,4,2, 1,5,6,7,8,2], s
        print('OK: BOS/EOS 位置逐位正确')
        "

    （`1`=BOS `2`=EOS；期望值是按 `starts`/`doc_starts` 手推出来的。如果你的实现给出
    别的结果，错的那张表几乎一定是 `doc_starts`。）

    再来两条：

        同 seed 两次顺序相同：BlockStream(100, 8, 42) 取 5 个 idx，重建一个再取 5 个，逐个相等。
        resume 等价：BlockStream(n, b, s) 连取 N+K 个 batch，与「取 N 个 -> state() ->
        新实例 load_state -> 再取 K 个」逐个相等。这条就是 C4 的最小版本，本文件必须自己先过。

    最后：max_blocks 语义 —— `build_packed(dir, block_size=128, max_blocks=0)` 与不传
    该参数结果相同（0 是「不限量」，见 config.py 注意 5）。
"""
