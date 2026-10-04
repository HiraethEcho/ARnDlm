"""utils/device.py -- 设备选择、dtype 名字到 torch.dtype 的转换、autocast 上下文。

用途
    这台机器是 AMD Radeon 880M + ROCm 10.0，torch 报的是 **cuda API**。
    本文件把「跑在哪、用什么精度」这两件事收在一处，别处不许自己挑。
    本文件**不 import 配置** —— 它只认传进来的 dtype 名字。`utils/config.py` 里那个
    `DTYPE` 字符串的读者是 `utils/loop.py`：训练时把 `config.DTYPE` 传进来，验证时硬传
    `"float32"`。同一个函数、两类调用方 —— 这个形状不是洁癖，是「跨实现比 loss 必须
    fp32」这条硬要求推出来的：如果 `autocast()` 自己去读配置，`eval_loss` 就没有办法
    关掉混合精度（实验版正是在 `common/loop.py:231` 硬传 `"float32"` 的）。

    对应 example/common/seeds.py 的**下半部分**（实验版 seed 和 device 同文件；
    实践版拆成 seed.py + device.py，理由写在 utils/seed.py 的「用途」里）。

要写什么
    四个模块级对象：

        _DTYPES = {"float32": ..., "bfloat16": ..., "float16": ...}

        pick_device(require_cuda: bool = True) -> torch.device
            不可用且 require_cuda 时抛 RuntimeError；否则**打印**选了什么和设备名。

        resolve_dtype(name: str) -> torch.dtype
            查 _DTYPES。未知名字必须响。

        autocast(device: torch.device, name: str = "bfloat16")
            上下文管理器。CPU 或 float32 -> contextlib.nullcontext()。

注意
    1. 宁可报错，不要静默退回 CPU。这台机器踩过的最坏一次：tensor 默认在 CPU，
       `torch.cuda.is_available()` 照样 True，日志一切正常，只是慢。所以
       require_cuda 默认 True，且选完要打印 —— 打印这一行是这条防线的全部。
    2. ROCm 上 `device_type` 仍然是 `"cuda"`。torch 走 HIP
       （`torch.version.hip` 有值、`torch.version.cuda` 是 None），但暴露的是 cuda API。
       写 `torch.autocast(device_type="hip")` 会报错。设备名会是
       "AMD Radeon 880M Graphics" 这种字符串 —— 别按 NVIDIA 的名字做判断。
    3. autocast 在 CPU 上必须是 nullcontext。不是 `torch.autocast("cpu", ...)`：
       CPU 的 bf16 autocast 有支持但行为不同，而本项目要在 CPU 上做逐元素对拍
       （example/teach/01 §1.6 的 logits max|Δ| ≤ 1e-5），CPU 路径必须严格 fp32。
    4. float32 也返回 nullcontext，而不是"开一个 fp32 的 autocast"。这条和 eval 绑在
       一起：跨实现比 loss 必须 fp32（bf16 的舍入差比要测的差异大，
       example/teach/00 §0.2 第 6 条）。所以 loop.py 会显式 autocast(device, "float32")
       来关掉混合精度 —— 本文件必须把那个调用变成 no-op，而不是报错。
    5. master weights 保持 fp32，只有 matmul 走 bf16。把 bf16 权重直接交给 optimizer，
       200 步看不出问题，10 万步之后小更新被舍入吃掉（example/teach/02 §2.7）。
       本文件不负责这件事，但 autocast 的边界画错就是它的表现，所以记在这里。
    6. resolve_dtype 用 `_DTYPES[name]`，不要用 `.get(name, torch.float32)`。
       兜一个默认值正是这一层要拦住的错误：配置写 `dtype: bf16`（拼错）应该当场
       KeyError，而不是安静地用 fp32 跑一遍，然后把吞吐差异解释成别的变量。

依赖
    torch、contextlib。被 utils/loop.py 消费（训练用配置里的 DTYPE，
    eval 强制 "float32"）。反过来**不** import utils/config.py —— 理由见上面「用途」，
    另外 config 层要在没有 GPU 的机器上也能 import，不能因为一个字符串转换就拉进硬件判断。

验收
    .venv/bin/python -c "from utils.device import pick_device, resolve_dtype, autocast; \\
        d = pick_device(); print(resolve_dtype('bfloat16'), resolve_dtype('float32')); \\
        import torch; x = torch.randn(4, device=d); \\
        with autocast(d, 'float32'): print('fp32 branch ok', (x + x).dtype); \\
        with autocast(d, 'bfloat16'): print('bf16 matmul ->', (x @ x.reshape(2,2).t()).dtype)"
        -> 打印 cuda 设备名 + bfloat16/float32；bf16 那次 matmul 的输出 dtype 是 bfloat16，
           而 x 本身仍是 float32（这就是「master weights 保 fp32」的可见证据）

    未知名字必须响：
    .venv/bin/python -c "from utils.device import resolve_dtype; resolve_dtype('bf16')"
        -> KeyError: 'bf16'   （不是 fp32）

    没有 GPU 的机器（或 require_cuda 传 True 而 CUDA 不可用时）：
    .venv/bin/python -c "from utils.device import pick_device; pick_device(True)"
        -> RuntimeError，且消息里写清了是哪一个条件没过
"""
from contextlib import nullcontext
import torch

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
def resolve_dtype(name: str) -> torch.dtype:
    return _DTYPES[name]

def pick_device(require_cuda: bool = True) -> torch.device:
    if require_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("有 CUDA 吗你就用")
        else:
            print("本训练有超级CUDA力")
            device = torch.device("cuda")
    else: 
        print("龟速 CPU")
        device = torch.device("cpu")
    print(f"[device] {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu only'})")
    return device

def autocast(device: torch.device, name: str = "bfloat16"):
    if device.type == "cpu" or name == "float32":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=resolve_dtype(name))
