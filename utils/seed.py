"""utils/seed.py -- 三处随机数源的设定，和给 resume 用的快照 / 恢复。

用途
    「这次运行可复现」是全部比较的前提。它由三件事组成：起点相同（seed_all）、
    中间状态可存（rng_state）、存了的能装回去且接得上（load_rng_state）。
    少任何一件，"同 seed 两次" 就只是一句口号。

    对应 example/common/seeds.py 的**上半部分** —— 实验版把 seed 和 device 写在同一
    个文件里，实践版按 utils/__init__.py 的清单拆成 seed.py + device.py。这是一次
    有意的偏离：代价是 A/B 时少了一个文件对文件的映射，收益是「设随机数」和
    「选硬件」这两件事的变更频率本来就不同（换机器只动 device.py）。
    device 那一半在 utils/device.py。

要写什么
    三个函数，全部模块级：

        seed_all(seed: int) -> None
            random.seed / np.random.seed / torch.manual_seed 三个都要。
            torch.cuda.manual_seed_all(seed) 可以直接调：**实测**（torch 2.12+rocm，
            `CUDA_VISIBLE_DEVICES=""`）无 GPU 时它不抛也不警告（`UserWarning` 提级到
            error 也没事），因为它是 lazy call。加 is_available() 判断也无害。

        rng_state() -> dict
            返回 {"python": ..., "numpy": ..., "torch": ...}，cuda 可用时再加
            "cuda": torch.cuda.get_rng_state_all()。

        load_rng_state(state: dict) -> None
            rng_state 的逆操作，键按存在与否取。

注意
    1. numpy 必须单独设。torch.Generator / torch.manual_seed 只管 torch 自己，
       而数据侧（未来的 sft 打包、A2D 的 corruption）可能从 numpy 抽数。少这一处
       的症状不是报错，是“第二次跑就不一样”。
    2. numpy 的函数名**带下划线**：`np.random.get_state()` / `np.random.set_state()`。
       `getstate` / `setstate`（numpy 1.x 的老名字）不存在，写错是 `AttributeError`，
       报错消息会提示 `Did you mean: 'set_state'` —— 看到就照它改。torch 的
       `get_rng_state` / `set_rng_state` 则是**有**下划线的，numpy 和 torch 两个命名
       习惯正好相反。
    2. seed_all 必须发生在 build_model **之前**。Loop.__init__ 里再 seed 已经晚了 ——
       参数在那之前就已初始化完。实验版第一版正是漏了这一步，"同 seed 同权重" 的
       检查当场失败（example/teach/01-hf-qwen.md §1.4）。这个文件只提供工具，
       调用时机由 models/hf_qwen/ 负责。
    3. 快照要四份齐：python / numpy / torch / cuda。少 numpy 那一份，resume 之后
       numpy 的序列会从头开始抽，而 loss 曲线只是"稍微对不上"，最难查的那类。
    4. load_rng_state 里 cuda 分支的条件是 `"cuda" in state and torch.cuda.is_available()`。
       两个都要：在有 GPU 的机器上存、在无 GPU 的机器上恢复（或反过来）不该炸。
       只写 is_available() 的话，恢复一份没有 "cuda" 键的旧快照会 KeyError。
    5. 恢复的必须是**训练循环结束时**那一份快照，不是"写 checkpoint 那一刻"的状态 ——
       中间的 eval 和采样会消耗随机数。实验版量过：取早了，resume 与直跑的逐步 loss
       差到 5.2e-01（example/common/loop.py:146-153 的注释）。那是 loop.py 的责任，
       但它只能靠本文件这两个函数做到。

依赖
    stdlib random、numpy、torch。注意：torch 在这个文件里出现是允许的，
    在 utils/config.py 里不允许 —— config 层要在没有 GPU 的机器上也能 import。

验收
    同 seed 两次序列相同：

        .venv/bin/python -c "import torch; from utils.seed import seed_all; \\
            seed_all(42); a=[torch.rand(3) for _ in range(2)]; \\
            seed_all(42); b=[torch.rand(3) for _ in range(2)]; \\
            print(all(torch.equal(x,y) for x,y in zip(a,b)))"
            -> True

    快照续接 == 一次跑完：先跑 5 个 rand 存 state，再跑 5 个；
    另一个进程 seed_all 后 load_rng_state(state)，接着跑 5 个 —— 两组逐个 torch.equal。

    故意漏一处看它是否响：把 np.random.seed 那行注释掉，第一条断言仍然 True
    （torch 不依赖 numpy 的种子）—— 所以这条测的是"它必须**不是**白测"：
    补一条用 np.random.rand 的断言，确认它会红。
"""
from __future__ import annotations
import random
import numpy as np
import torch

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)          # 同时管 CPU 和所有 CUDA device
    torch.cuda.manual_seed_all(seed) # 冗余但无害

def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

def load_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
