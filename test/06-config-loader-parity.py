#!/usr/bin/env python
"""test 06 -- 实践版的加载器和实验版的加载器行为一致吗，config.py 报错够响吗。

用途
    两个断言，都很便宜，都盯着「看不见失败」的那一层：

      A. 解析同源。utils/version.py::parse_yaml 是
         example/common/version.py::parse_yaml 的第二个实现。同一个解析器写两遍，
         正是 configs/README.md 警告的那条「两条代码路径」—— 这个文件就是为它付的账。
         同样的文本进去，必须得到完全相同的 dict 出来；否则实践版跑的配置和实验版跑的
         不是同一份，比较立刻失去意义。

      B. 未知键要响。utils/config.py::_from 必须报错，并且在**一条消息里列出全部**
         未知键。一次只报一个是 `cls(**raw)` 的 TypeError —— 那正是要禁止的形状。

    为什么 B 和 A 放一起：两条都在说配置层的沉默。解析错的标量和被吃掉的键，在 loss
    曲线上长得一模一样。

要写什么
    一个独立脚本，跟 test/01..05 同形状 —— 不用 pytest，打印 PASS/FAIL，失败时
    `sys.exit(带消息)`。大致：

        ROOT = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(ROOT))              # 让 `import utils` 能用
        sys.path.insert(1, str(ROOT / "example"))  # 让 `import common` 能用

        from utils.version import parse_yaml as mine
        from common.version import parse_yaml as theirs
        from utils import config as C

    Part A：把 configs/ 下的每个文件都读一遍（example/ 和 practical/ 都要），两个解析器
    各跑一次，用 `==` 比。同时确认 dataclass 需要的那些键确实都在。

    Part B：复制一份真实 `train:` 段的 dict，塞进假字段，在 try/except 里调
    C._from(raw, C.TrainCfg)，断言 (a) 报错了，(b) 塞两个时消息里**两个名字都在**。

    Part C -- 解析器为什么手写，写成可检查的形式。pyyaml 在 .venv 里装着（6.0.3），
    而 YAML 1.1 的浮点正则要求有小数点：`3e-4`、`1e-6` 回来是字符串，`1.0e-6` 才是
    float。所以直接从 utils/config.py 断言每个数值字段真的是数 ——
    `type(C.AR_TRAIN.lr) is float`，min_lr_ratio 同理，`type(C.TIER0.rms_norm_eps) is
    float` 同理。哪天有人换了 safe_load 又没重写配置里的指数，报错落在这里，而不是 300
    行之后掉进 AdamW 里。顺手把一个字段在 pyyaml 下的值打印在你的值旁边，让分歧看得见，
    而不是被断言默默吃掉。

注意 —— 这个测试顺便要练到的 Python 细节
    * `sys.path` 的插入必须发生在需要它的 import **之前**，且在模块顶层。放在函数里，
      文件顶部那些 import 早就失败了。
    * `example` 不是一个包 —— `example/common/` 才是，而且它以 `common` 的名字被 import，
      因为 train_ar.py 自己插的是 `example/`
      （`sys.path.insert(0, str(Path(__file__).resolve().parents[2]))`）。所以插
      `example/`，不是仓库根，否则 ModuleNotFoundError: common。
    * 两个包里同名的 `version` 模块是两个不同对象，`mine` / `theirs` 各自绑定得很干净。
      但如果图省事只 import 一个、假设另一个也能从同一个名字拿到，你就会拿一个模块和自己
      比 —— 那条断言从此永远通过。先检查 `mine is not theirs`；这句是后面一切成立的前提。
    * dict 的 `==` 对嵌套 dict 是递归的，所以 `{"train": {"lr": 0.0003}} == {...}` 已经
      覆盖到你在意的深度，不用再自己走一遍。
    * Python 里 `1024 == 1024.0` 为真，所以一个把 steps 转成 float 的解析器能通过 `==`。
      标量的 `type(...)` 也要比，否则 A 恰恰在最可能发生的那种 bug 上「不可能失败」。
    * `try/except Exception as e` 然后取 `str(e)` —— B 断言的是消息**内容**，不只是异常
      类型，因为 B 关心的就是响不响。

依赖
    utils/version.py、utils/config.py、configs/example/tier0.yaml、
    configs/practical/smoketest.yaml。不需要 torch、不需要 GPU、不需要语料。

验收
    .venv/bin/python test/06-config-loader-parity.py
    -> 退出码 0，打印比对了多少个文件、以及被塞进去的字段名。

    当着你的面故意让它红一次：把其中一个解析器改成 float 先于 int 转换，确认测试失败。
    一个从没见它失败过的测试不算证据。

    顺便手看一眼：`cell_path('x', kind='chk')` 返回的是 logs/run 路径而不是报错。这是从
    example/common/version.py 继承的行为 —— 为了同源先留着，但要知道这个拼写错误是沉默的。
"""
