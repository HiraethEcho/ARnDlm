# Python 与 config 层: 七个易错细节

读 `example/`、写 `utils/` 时撞到的问题。共同特征: **不报错, 只是结果不对** ——
或者语法合法、分支永不执行。

引用格式 `file::symbol`。**不记行号**, 本仓库在活跃编辑中, 行号会漂移。

---

## 1 `from __future__ import annotations`

**做什么**: 让本文件的注解延迟求值。

| | `f.__annotations__["x"]` |
|---|---|
| 加这行 | 字符串 `"Path"` (PEP 563) |
| 不加, Python 3.14+ | 惰性求值函数 (PEP 649/749) |
| 不加, Python ≤ 3.13 | 类型对象本身 |

**作用**: ① 前向引用不必加引号 ② 注解里的表达式不在 import 时构造
(如 `dict[str, list[int]]`)。

**为什么必须放最顶部**: `__future__` 是编译期开关。`from __future__ import X`
不 import 任何运行时对象, 它设的是 code object 的 flag (`CO_FUTURE_ANNOTATIONS`)。
改的是编译行为, 所以位置限制比普通 import 严。

**在本仓库**: Python 3.14 默认已延迟求值, 这行基本是样板。差别只剩
`__annotations__` 拿到的是字符串还是惰性值, 没有任何代码依赖它。

**本仓库的实际分布** (实测 69 个 py 文件):

| 范围 | 情况 |
|---|---|
| `example/` 全部模块 | 都有 (5 个 `__init__.py` 除外) |
| `utils/config.py`、`utils/version.py` | 都有，且都放在长 docstring **之后** |
| `utils/data.py`、`device.py`、`seed.py` | **没有** —— 这三份没有自引用注解, 3.14 上不加也不报错 |

放的位置值得注意: `utils/` 两个文件把长 docstring 写在最前面, 这行跟在后面 ——
**这才是合法顺序**。future statement 之前只允许出现这几种行: 模块 docstring、注释、
空行、其它 future statement。所以"必须在第 1 行"是常见误记; 准确说法是
**必须在模块 docstring 之后、任何代码之前**。反过来（future 写在 docstring 前面）
报 `SyntaxError`。

**实测 (Python 3.14.7)**: 类体内写自引用注解 `def me(self) -> A`（`A` 就是本类）、
且**不加** 这行, import 照常成功 —— PEP 649 把注解求值推迟了。所以这一行在本仓库
确实是样板, 它的价值在 ≤3.13 上（那里不加就是 `NameError`）。

---

## 2 YAML 解析: 用包还是手写

**结论**: 能用包。`pyyaml` 在 `requirements.txt` 里已声明
(`pyyaml  # configs/*.yaml`), 且已装在 `.venv`。
"没有 pyyaml 所以只能手写" 这条理由不成立。

**但直接换 `yaml.safe_load` 会静默改类型。**

成因: PyYAML 实现 YAML 1.1, 它的 float resolver 正则要求数字里**必须有小数点**。
`1e-6` 没有点 → 不是 float, 也不是 int → 落到 **str**。

| 文本 | 手写 `_scalar` | `yaml.safe_load` |
|---|---|---|
| `rms_norm_eps: 1e-6` | `float` | **`str`** `'1e-6'` |
| `lr: 3e-4` | `float` | **`str`** `'3e-4'` |
| `dlm_lr: 1e-4` | `float` | **`str`** `'1e-4'` |
| `rope_theta: 10000.0` | `float` | `float` |
| `steps: 1024` | `int` | `int` |
| `packing: true` | `bool` | `bool` |
| `null` / `~` | `None` | `None` |

后面四个是对齐的 —— 有小数点的浮点、整数、bool、null 在两边一致。
**不齐的只有"无小数点科学计数法"这一类。**

**后果分两种, 都不报错**:

1. `lr * schedule` → `TypeError`。至少能看到。
2. f-string 格式化后写进 `summary.json` → `"lr": "3e-4"`, 无异常。
   **这是危险的那种。**

**修法** (选改动最小的):

- `lr: 3.0e-4` —— 加点, 只改 config
- `lr: !!float 3e-4` —— 显式 tag
- 读出来强制 `float(cfg["train"]["lr"])` —— 只改代码

**其它 YAML 1.1 隐式类型坑** (本仓库 config 目前一个都没用到):

| 输入 | YAML 1.1 结果 |
|---|---|
| `yes` / `no` / `on` / `off` | `bool` (Norway problem) |
| `1:30` | `90` (sexagesimal) |
| 重复 key | 静默 last-wins |

**注意方向**: 手写 `_scalar` 在这里是 YAML 1.1 的**宽松超集** ——
它把 `3e-4` 认成 float, PyYAML 不认。宽松版掩盖了 config 的写法问题。

**包的选择**:

| 方案 | 类型规则 | 代价 |
|---|---|---|
| `yaml.safe_load` (PyYAML) | YAML 1.1, 无点科学计数 → str | 已装, 换 1 行 |
| `ruamel.yaml` | YAML 1.2, `1e-6` 就是 float | 新依赖 |
| `tomllib` (stdlib, 3.11+) | TOML, 规则严格 | config 全部改写 |
| `json` (stdlib) | 无注释 | config 的注释会丢, 不合适 |

**换包前必须先决定的**: 如果 `example/` 和 `utils/` 两份 loader 必须保持
"两份独立实现 + 断言相等" (见 `test/06-config-loader-parity.py`),
则只换一边会让该断言变红, 两边都换会让该断言失去意义。

---

## 3 `str.strip()`

去掉首尾空白。空白的定义是 `str.isspace()` 为真的字符:
空格、`\t`、`\n`、`\r`、`\v`、`\f`、`\xa0`。返回**新字符串**, 无原地版本。

**坑 1 —— 不带括号是 bound method, 恒真**:

```python
if not s.strip:     # ✗ 绑定方法对象, 永远为真 -> 死分支
if not s.strip():   # ✓
```

少一对括号, 判空检查变成永不执行的代码, 语法不报错。

**坑 2 —— 参数是字符集, 不是前后缀**:

```python
"cat".strip("at")          # 'c'  —— 不是去掉 "at" 这个子串
"cat".removeprefix("at")   # 'cat'  (3.9+)
```

**在栈式 `parse_yaml` 里 4 处用法, 只有 2 处承重**:

| 位置 | 承重? | 原因 |
|---|---|---|
| `if not line.strip():` | **冗余** | 上一行 `.rstrip()` 已保证 `line` 要么是 `""`, 要么以非空白结尾 |
| `len(line) - len(line.lstrip())` | 承重 | 数缩进宽度 |
| `line.strip().partition(":")` | **冗余** | 后面 `key.strip()` 会修掉 |
| `key.strip()` / `value.strip()` | 承重 | `partition(":")` 保留冒号两边的空格 |

`partition(":")` 的行为: 在**第一个**冒号处切成 3 段, 找不到冒号时是
`(line, "", "")`。所以值里的冒号没事 (`path: s3://x` → `'s3://x'`),
key 里的冒号不行。

---

## 4 栈式缩进解析器

**原理**: YAML 用**缩进**表达嵌套。所以解析 = 维护一个"当前打开着的父节点"栈,
逐行扫, 缩进回退就弹栈。

**结构**: `stack` 元素是 `(indent, dict)` 二元组。

| 字段 | 含义 |
|---|---|
| `indent` | 打开这个 dict 的**那一行自己**的缩进宽度 |
| `dict` | 那个 dict |

栈底压哨兵 `(-1, root)`。`root` 是最终返回值。

**逐行**:

1. `line = raw.split("#", 1)[0].rstrip()` —— 切注释, 去尾空白。`1` = 只切第一刀。
2. 空行/纯注释行 → `continue`。
3. `indent = len(line) - len(line.lstrip())` —— 存**字符数**, 不是层级号。
   YAML 不要求固定 2 空格。
4. `key, _, value = line.strip().partition(":")` —— 第一个冒号处切 3 段。
5. `while stack[-1][0] >= indent: stack.pop()` —— 弹掉所有同级或更深的层。
6. `parent = stack[-1][1]` —— 栈顶就是本行的父。
7. `value.strip() == ""` → 这是**父节点行**。建 child, 写进 parent,
   把 `(indent, child)` 压栈。
8. 否则 → 叶子标量, `parent[key] = _scalar(value)`。

**走一遍**:

```yaml
mode: ar

optim:
  lr: null
  grad_clip: null

schedule:
  warmup_steps: null
```

```
line                   indent  pops  stack after           action
mode: ar                    0     0  [-1]                  mode = 'ar'
optim:                      0     0  [-1, 0]               push child 'optim'
lr: null                    2     0  [-1, 0]               lr = None
grad_clip: null             2     0  [-1, 0]               grad_clip = None
schedule:                   0     1  [-1, 0]               push child 'schedule'
warmup_steps: null          2     0  [-1, 0]               warmup_steps = None
```

两个关键动作:

- **`schedule:` 弹 1 次。** 它缩进 0, 栈顶 `optim` 也是缩进 0, `0 >= 0` 成立 → 弹。
  **这就是第 5 步必须用 `>=` 而不是 `>` 的原因。** 写成 `>` 时 `optim` 不会被弹,
  `schedule` 会变成 `optim` 的子键 —— 然后 `TrainCfg` 报 TypeError, 错误现场离 bug 很远。
- **`lr: null` 弹 0 次。** 缩进 2, 栈顶缩进 0, `0 >= 2` 不成立 → parent 就是 `optim`。

**两个设计决定**:

- **哨兵用 `-1`。** 真实缩进恒 ≥ 0, 所以 `-1 >= indent` 恒假 → 哨兵永不被弹 →
  栈永不为空 → `stack[-1]` 不必判空。参考实现里的 `while stack and ...` 在这个
  不变式下不可达。
- **存的是"开这层那一行"的缩进, 不是子行的。** 弹栈判据是"谁开的我",
  所以压栈时必须用本行的 `indent`。

**弱点** (docstring 只写了 list 那条, 其余实测):

| 输入 | 结果 | 说明 |
|---|---|---|
| `items:\n  - a\n  - b` | `{'items': {'- a': {}, '- b': {}}}` | 不支持 list, 把 `- a` 当 key |
| `a:\n   b: 1\n  c: 2` | `{'a': {'b': 1, 'c': 2}}` | **缩进不一致不报错**, 静默改结构。PyYAML 抛 `ParserError` |
| `purpose: "a #1 test"` | `{'purpose': '"a'}` | 切 `#` 不看引号 |
| `justtext` (整行无冒号) | `{'justtext': {}}` | 静默建了个空 child |
| `a: 1\na: 2` | `{'a': 2}` | 重复 key last-wins, 无警告 |

第 2 条最危险: 第 5 步只比较数字大小, **没有"同一层必须同缩进"的校验**。

**和 `_scalar` 的耦合**: 类型判定顺序 **bool → null → 引号串 → int → float → str**
不能调换。

- `float()` 提前 → `steps: 1024` 变 `1024.0`
- `true`/`false` 没被第一步截住 → 变成字符串 `"false"`, 非空 →
  `if tie_word_embeddings:` 对一个写着 false 的 config 恒真

---

## 5 tied embedding 重复进 optimizer

**tied 的含义**: `lm_head.weight` 与 `embed_tokens.weight` 是**同一个 tensor 对象**,
共享 storage。不是"两个值恰好相等"。

**取参数的三种方式**:

| 取法 | 得到的个数 |
|---|---|
| `model.parameters()` | 1 (默认 `remove_duplicate=True`) |
| `model.named_parameters()` | 1 |
| `model.named_parameters(remove_duplicate=False)` | 2 |
| `model.state_dict()` | 2 个键 |

所以按 `id()` 去重是**保险**, 在默认取法下不生效 —— 但它防的是别人改成
`remove_duplicate=False` 之后的情形。

**重复传的后果** (实测, 常数梯度, AdamW, 20 个 iteration):

| 配置 | 结果 (展平后第 0 个元素) |
|---|---|
| 单条, 20 步, `lr=1e-3` | `-0.13258` |
| **重复传两次**, 20 步, `lr=1e-3` | `-0.15258` |
| 单条, **40 步**, `lr=1e-3` | `-0.15258` |
| 单条, 20 步, **`lr=2e-3`** | `-0.15258` |

- **重复 20 步 == 单条 40 步, 逐位相等** (`allclose(atol=0, rtol=0)` → `True`)
- **重复 20 步 != lr 翻倍**, 逐位 `False`, 只差 ~1e-8 的浮点舍入
- 位移量 (`n=16` 元素的 Frobenius 范数): 单条 20 步 = `0.08`,
  后三者都是 `0.16`

**结论: 精确说法是"等效步数翻倍", 不是"等效学习率翻倍"。** lr 翻倍只是位移量
相同, 浮点路径不同。

**机制两条**:

1. `optimizer.state` 按 **tensor 对象**做 key。重复传时只有 **1 份** state,
   两个条目共用 → `exp_avg` / `exp_avg_sq` 每次 iteration 被推进**两次**。
2. `state['step']` 也共用 → 20 个 iteration 后计数是 **40**。

所以单条 40 步是 `(ema, update) × 40`, 重复 20 步是
`((ema, update)(ema, update)) × 20` —— 完全同一串动作, 因此逐位相等。

**三个容易漏掉的点**:

1. **PyTorch 会警告, 但现在不报错**:

   ```
   UserWarning: optimizer contains a parameter group with duplicate parameters;
   in future, this will cause an error; see github.com/pytorch/pytorch/issues/40967
   ```

   当下只看到一条 warning, 训练照跑、loss 照降。"in future" 意味着以后会变硬错误。

2. **`state_dict()` 那条路连 warning 都没有**。这是最难发现的一种:

   ```
   sd["emb.weight"] is sd["head.weight"]        -> False   # 不是同一对象
   data_ptr 相同                                 -> True    # 但同一块显存
   list(sd.values()) 交给 AdamW                  -> (无警告)
   ```

   `state_dict()` 默认 `keep_vars=False`, 返回的是 `param.detach()` ——
   **新建的 Tensor 对象, 共享 storage**。AdamW 看到两个不同对象, 于是建
   **两份独立 state**, 但两次更新写进同一块显存。更新照样翻倍, 重复检测抓不到。
   它的 `requires_grad` 也是 `False`。

3. **梯度累加是正确行为, 别一起改**。反向传播时两条路径的梯度累加到同一个
   `.grad`, 这是 tied 权重应有的。出问题的**只在 optimizer 层面的重复更新**。
   `weight_decay` 也跟着翻倍, 因为它在 optimizer 里做。

**症状**: loss 曲线看起来收敛更快, 实际是把训练跑了 2 倍步数。所有基于 step
的调度 (warmup / cosine) 整体错位, 续训和复现都对不上。

**最简复现**:

```python
p = nn.Parameter(torch.randn(4, 4) * 0.1)
opt = torch.optim.AdamW([{"params": [p, p], "weight_decay": 0.0, "lr": 1e-3}])
for _ in range(20):
    opt.zero_grad(); p.grad = torch.full_like(p, 0.5); opt.step()
# 结果 == 同样配置但 [p] 跑 40 步
```

---

## 6 评测 prompt 必须固定

`config.py` 里的 `PROMPTS` 属于**评测几何**, 是代码常量, 不进 YAML。
`utils/config.py` 的模块级常量清单把它和 `EVAL_BATCHES` 并列, 注释是
"评测几何, 留在代码里"。

**为什么必须两版一致**:

1. **它是固定输入, 不在变量表里。** 项目按 G1-G5 五组组织, 每组只动一个变量。
   prompt 不是变量 → 换掉它, 两版的 `logs/run/*/samples` 就不能并排看,
   G1 ("两种 AR 写法是不是同一个模型") 少一条定性证据。
2. **默认参数绑定, 改一处影响全部路线。** 采样函数都写成
   `def sample_*(model, prompts=PROMPTS, ...)`, 训练脚本基本不传这个参数。
3. **没有任何保护网。** YAML 层的校验管不到它, 也没有测试引用它
   (`grep PROMPTS test/` → 0 命中)。docstring 是唯一的契约, 契约和实现分叉时
   没有东西会提醒。

**什么不受影响**: `PROMPTS` 只喂给采样函数, 不进训练。
loss、复现性、数值对拍都不受影响 —— 对拍用 `torch.randint(..., generator=...)`
造随机输入, 不用 prompt。

**两套 prompt 不是不一致**: `example/base/hf/model.py::SFT_PROMPTS` 是中文对话
prompt, `train_sft.py` / `train_dpo.py` / `train_dlm_sft.py` 用它是设计如此。
pretrain 用英文 prompt、SFT 用中文 prompt 是对的。

**要改评测 prompt 时**: **追加, 不替换**。替换会让之前所有 run 的 samples
变成不可比, 追加只是多一列。`SPEC.md` §6.4 也要求正式评测用
"几百 token × ≥5 个 prompt", 本来就要扩。

---

## 7 `@` 装饰器语法

**`@` 是装饰器的语法糖, 语义是"调用 + 重新绑定"**:

```python
@X
def f(): ...
```
≡
```python
def f(): ...
f = X(f)
```

`@` 后面可以是任意表达式, 只要求值出来是 callable。
`@dataclass(frozen=True)` 是**两层调用**: 先 `dataclass(frozen=True)` 拿到包装函数,
再拿包装函数套 `ModelCfg`。这就是"带参数的装饰器"的全部含义, 没有额外语法。

多个装饰器**从下往上**套: `@a` `@b` `def f` → `f = a(b(f))`。

**`@` 本身没有语义**, 语义在被调用的东西上:

| 写法 | `@` 后面 | 干什么 |
|---|---|---|
| `@dataclass(frozen=True)` | 标准库函数 | 按注解生成 `__init__`/`__repr__`/`__eq__`; 禁止赋值 |
| `@property` | 内置类 `property` | 方法变属性, `obj.x` 自动调 `fget(obj)` |
| `@staticmethod` / `@classmethod` | 内置类 | 改变第一个参数的绑定方式 |
| `@functools.cache` | 标准库函数 | 套一层缓存 |
| 你写的任意函数 | 函数 | 行为完全由你决定 |

**和 dataclass 的配合**: dataclass 只把**带注解的类属性**当字段。
`@property` 修饰的方法没有 `head_dim: int` 这样的注解, 所以它
① 不进 `__init__` 的参数表 ② 不是 `fields(cls)` 的成员。

第二条有实际效果: 用 `{f.name for f in fields(cls)}` 算允许的键时, `head_dim`
不在里面, 于是 YAML 里写 `head_dim: 64` 会被判未知键并报错。
**这是正确行为** —— 它是派生值, 写进 YAML 就有机会和 `hidden_size` / `num_heads`
互相矛盾。所以 property 让它"只能算、不能设"。

调用时 `self` 由 property 协议自动传: `m.head_dim` 实际执行
`ModelCfg.head_dim.fget(m)`。所以**不能写** `m.head_dim()`。

---

## 8 RNG 快照 API: numpy 和 torch 的下划线规则相反

最容易写错的一对: 看起来对称, 拼写习惯却不同。

| 库 | 取当前状态 | 装回状态 |
|---|---|---|
| numpy (>=1.17) | `np.random.get_state()` | `np.random.set_state(state)` |
| numpy (1.x 老名字, 已移除) | `getstate` ✗ | `setstate` ✗ |
| torch | `torch.get_rng_state()` | `torch.set_rng_state(state)` |
| torch, cuda 多卡 | `torch.cuda.get_rng_state_all()` | `torch.cuda.set_rng_state_all(states)` |

写错老名字的报错自带答案:
```
AttributeError: module 'numpy.random' has no attribute 'setstate'.
Did you mean: 'set_state'?
```

返回类型不同但都能被 `torch.save` 序列化, 所以可以一起塞进同一个 `train_state.pt`:
`np.random.get_state()` 返回 tuple, `torch.get_rng_state()` 返回 ByteTensor。

## 8b CUDA 的 lazy 调用: 无 GPU 时不抛也不警告（实测）

torch 2.12.0+rocm10.0.0, `CUDA_VISIBLE_DEVICES=""`:

| 调用 | 结果 |
|---|---|
| `torch.cuda.manual_seed_all(42)` | 不抛, 无 warning（`-W error::UserWarning` 也过） |
| `torch.cuda.set_rng_state_all([...])` | 不抛 |

原因: 走 `torch.cuda._lazy_call`, CUDA 未初始化时把回调排队, 于是变成 no-op。

所以两条常见说法要改:

- 「无 GPU 的机器上必须用 `if torch.cuda.is_available()` 包一层, 否则炸」—— **本 torch 版本下是错的**
- 反过来也不能据此认为 cuda RNG 一定被覆盖了。真正要确认的是**有 GPU 时**
  `get_rng_state_all()` 非空

实践上两种写法都能用, 挑一种别混: 总是写 `"cuda"` 键（不可用时存 `None`），
或者不可用时干脆不写这个键（`example/common/seeds.py` 是后者, `utils/seed.py` 是前者）。
两者恢复路径的判据要跟着各自的形式写（`is not None` vs `"cuda" in state`）。
