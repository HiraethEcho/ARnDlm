"""utils/version.py -- 把 configs/<version>/<tier>.yaml 读成 dict，并决定一个格子的输出目录。

用途
    配置栈的最底层。只管「路径 + 文本」，不含任何实验数字 —— 数字属于
    configs/practical/*.yaml，通过这里读进来。一个数字出现在本文件里，就说明它放错了地方。

    对应 example/teach/00-common-base.md 的 Step 0.1。utils/ 下每个文件都会 import 它，
    所以它是实践版第一个该写的文件。

要写什么
    四个模块级函数，不用类：

        ROOT          Path   # 仓库根 = Path(__file__).resolve().parents[1]
        CONFIG_ROOT   Path   # ROOT / "configs"

        config_path(version: str, tier: str) -> Path
            CONFIG_ROOT / <version> / f"{tier}.yaml"
            `tier` 就是文件名：load_config("practical", "smoketest") 打开的是
            configs/practical/smoketest.yaml。文件里那个 `tier:` 字段没有任何代码读它，
            它只是快照。两者不一致时的症状是 FileNotFoundError，而你会去查错的地方。

        load_config(version: str, tier: str) -> dict
            文件不存在就抛错，并把缺的路径写清楚。不要返回 {}：那样调用方会死在
            config.py 里的一个 KeyError 上，于是你去 debug config.py。

        parse_yaml(text: str) -> dict
            缩进嵌套映射、标量、`#` 注释。**不**支持 list、多行字符串、锚点 ——
            配置文件里没用到这些。

        cell_path(name: str, version: str = "practical", kind: str = "ckpt") -> Path
            kind == "ckpt" -> ROOT / "checkpoints" / f"{name}{suffix}"
            kind == "run"  -> ROOT / "logs/run"   / f"{name}{suffix}"
            后缀表 {"example": "-example", "practical": ""}。
            后缀只在这一个地方拼。调用点自己拼名字就会拼错，而这里「错」的意思是把东西
            写进另一个版本的目录 —— 不报错，是一次覆盖。
            实践版后缀为空 => cell_path('ar-512-smoke') = checkpoints/ar-512-smoke。
            共享的 checkpoints/ 里现有 20 个目录全部以 -example 结尾，所以实践版的名字
            撞不到实验版格子；撞得到的是 **main worktree 跑出来的实践版格子**（同一个
            /mnt/extra/ckpt，同样空后缀）。
            `kind` 也是个裸三元：不等于 "ckpt" 的一律当 logs/run，所以 kind="chk" 会把
            权重静默写进日志树。

注意
    1. 先剥 `#` 再取值。顺序是 line.split("#", 1)[0].rstrip()，然后判空行。反过来写，
       值里会留着 "# not this step" 当文本。
    2. 标量转换顺序：bool -> null -> 引号串 -> int -> float -> 原样 str。float() 排在
       int() 前面，steps: 1024 会变成 1024.0。true/false 没在第一步接住就变成字符串
       "false"，非空字符串恒真 => 一份写着 tie_word_embeddings: false 的配置在
       `if tie_word_embeddings:` 下面变成 true。
    3. 出栈条件是 `stack[-1][0] >= indent`。写成 `>` 的话同级的兄弟 key 会改成嵌套：
       batch_size 底下挂出一个 seed，TrainCfg 于是少字段。TypeError 报在 config.py，
       而 bug 在这里 —— 你会去查 config.py。
    4. 为什么手写解析器而不是 pyyaml（.venv 里装着 6.0.3，所以「没装」不是理由）：
       a. 同源。example/common/version.py::parse_yaml 是对照的另一边。把这边换成第三方
          解析器，「两份配置一致」就变成不可证伪的了 —— test/06 存在的意义就是这条。
       b. YAML 1.1 的浮点正则要求有小数点，pyyaml 照它办事：`3e-4` 和 `1e-6` 回来是
          **字符串**，`1.0e-6` 才是 float。本仓库的配置写成 `3e-4` 这种形式，所以此处
          手写的那个才是算对的一方。
       真需要 list 的时候别长大这个解析器：加 pyyaml 的同时，在**同一个 commit** 里把
       configs/ 里每个指数都补上小数点。
    5. 加载器同源约定（2026-10-05 定）：本文件是第二个实现，所以
       test/06-config-loader-parity.py 必须断言它与 example 那份对同一文本结果相等。
       没有那条断言，本文件就是 configs/README.md 警告的那条「两条代码路径」。
    6. 出栈那行写成 `while stack[-1][0] >= indent:`（没有 `stack and` 保护）是安全的，
       但安全性来自哨兵 -1：indent 恒 >= 0 > -1，栈永远弹不空。改了那个 -1 就会
       IndexError —— 动之前先回来看这条。

依赖
    只用标准库 pathlib。被 utils/config.py、utils/data.py 和每个训练入口 import。

验收
    .venv/bin/python -c "from utils.version import load_config as L; \
        t = L('practical', 'smoketest')['train']; \
        print(t['lr'], t['min_lr_ratio'], type(t['steps']).__name__)"
        -> 0.0003 0.1 int

    .venv/bin/python -c "from utils.version import ROOT, cell_path as P; \
        print(ROOT); print(ROOT.is_dir() and (ROOT / 'configs').is_dir()); \
        print(P('ar-512-smoke'))"
        -> ARnDlm-learn 的根目录、True、checkpoints/ar-512-smoke

    .venv/bin/python test/06-config-loader-parity.py
"""

from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs"
EXPERIMENT_SUFFIX = "-example"
VERSION_DIR = {"example": "example", "practical": "practical"}
SUFFIX = {"example": EXPERIMENT_SUFFIX, "practical": ""}

def config_path(version: str, tier: str) -> Path:
    if version not in VERSION_DIR:
        raise ValueError(f"version must be one of {sorted(VERSION_DIR)}, got {version!r}")
    return CONFIG_ROOT / VERSION_DIR[version] / f"{tier}.yaml"

def load_config(version: str, tier: str) -> dict:
    path = config_path(version, tier)
    if not path.exists():
        raise FileNotFoundError(
                f"{path} not found.\n"
                f"  go back and write the config file, or check the strings you gave me:\n"
                f"  version: {version}   tier: {tier}")
    return parse_yaml(path.read_text())

def parse_yaml(text: str) -> dict:
       # why i have to impl a parse for yaml myself? why not just use a package?
       # assistant: 两条理由，第一条是实测。
       #   1) YAML 1.1 的浮点正则要求有小数点，pyyaml 照它办事：
       #        yaml.safe_load('a: 3e-4')   -> {'a': '3e-4'}    # str
       #        yaml.safe_load('b: 1e-6')   -> {'b': '1e-6'}    # str
       #        yaml.safe_load('c: 1.0e-6') -> {'c': 1e-06}     # float
       #      smoketest.yaml 用的是 3e-4 / 1e-6 写法，换 pyyaml 后 lr 和
       #      rms_norm_eps 都是字符串，一路走到 AdamW / 模型里才炸。
       #   2) test/06 断言的是「两个手写解析器逐键相等」。换掉其中一边，
       #      这条断言就没有对象了。
       #   pyyaml 6.0.3 在 .venv 里装着 —— docstring 里「.venv 没有 pyyaml」那句是错的，我已经改过；你可以按自己判断再改回去
    root: dict = {}
    stack = [(-1, root)]
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        while stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        key, _, value = line.strip().partition(":")
        if value.strip() == "":
            child: dict = {}
            parent[key.strip()] = child
            stack.append((indent, child))
        else:
            parent[key.strip()] = _scalar(value.strip())
    return root

def _scalar(text: str):
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    if text.lower() in ("null", "none", "~"):
        return None
    if text.startswith(("'", '"')) and text.endswith(("'", '"')):
        return text[1:-1]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text

def cell_path(name: str, version: str = "practical", kind: str = "ckpt") -> Path:
    if version not in SUFFIX:
        raise ValueError(f"wtf are you doing? version in {sorted(SUFFIX)}!")
    root = ROOT / "checkpoints" if kind == "ckpt" else ROOT / "logs/run"
    return root / f"{name}{SUFFIX[version]}"
