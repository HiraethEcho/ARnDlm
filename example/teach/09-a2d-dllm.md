# Step 9 — `training/a2d_dllm/`

**你要写的**：`training/a2d_dllm/`（转换 + 训练 + 采样都走 dllm 包）
**A 版参考**：`example/base/hf/a2d_dllm.py`
**验收**：**C5** —— 转换后 0 missing / 0 unexpected / 0 改动；**C2** —— loss 对得上。

```sh
.venv/bin/python example/check_ab.py all
```

---

## 9.1 这一格的意义：走通上游，不重写上游

A 版**没有**重新实现 dllm 的 loss。三个部件全部来自包：

| 部件 | dllm 位置 |
|---|---|
| A2D 模型类 | `dllm/pipelines/a2d/models/qwen3/modeling_qwen3.py` |
| 转换 | `dllm/pipelines/a2d/convert.py` 的逻辑 |
| loss | `dllm/core/trainers/mdlm.py::MDLMTrainer.compute_loss` |
| scheduler | `dllm/core/schedulers/alpha.py::LinearAlphaScheduler` |
| 采样 | `dllm/core/samplers/mdlm.py::MDLMSampler` |

**抄一遍 dllm 的 loss 再叫它「用包」是没有意义的** —— 那测的是你的抄写。

## 9.2 不构造 Trainer，直接调 compute_loss

`MDLMTrainer` 是 `transformers.Trainer` 的子类，构造它要 dataset、
真 tokenizer、output_dir 脚手架 —— 这些都不在被测代码路径上。

A 版的做法：

```python
trainer = MDLMTrainer.__new__(MDLMTrainer)
trainer.scheduler = LinearAlphaScheduler()
trainer.time_epsilon = 1e-3
trainer.loss_weight_type = "scheduler"
trainer.loss_norm_type = "token"
trainer.right_shift_logits = False
trainer.processing_class = _ProcessingStub()   # padding_side / mask_token_id / bos / eos
trainer.meter = _MeterStub()                   # update(**kw) -> None
loss = trainer.compute_loss(model, {"input_ids": batch, "labels": batch})
```

`compute_loss` 只读这 6 个属性 + `self._preprocess/_postprocess/_compute_loss_weights`。

**这是个取舍**：省掉了 Trainer 的初始化开销，代价是绕过了 Trainer 的官方用法。
正式项目里应该先试正常构造，失败再退回这里，并在注释里写清为什么。

## 9.3 上游的 patch 是这个项目的一部分

`dllm` 的三个 A2D 类都没定义 `config_class`，而 4.57 的
`AutoModel.register` 新增了一致性校验（`auto_factory.py:624`）→
`import dllm.pipelines.a2d` **在模块顶层就炸**。

修法：`scripts/patches/dllm-0001-a2d-config-class.patch`。
A 版能 import 就是因为这个 patch 已经打上了。

**B 版写 `training/a2d_dllm/` 时也要确认 patch 在**：
`git -C ~/.cache/ARnDlm/dllm diff --cached --stat`

## 9.4 dllm 的 A2D 模型是怎么关掉因果的

不是改一个 flag，而是**整个替换 `forward`**：把 `create_causal_mask`
那段注释掉，改成「没有 mask 就当成全部有效」的 2D mask。

所以：

- 传 `attention_mask=None` → 双向，全位置可见
- 传 2D mask → 转成 4D，只有 padding 被屏蔽
- **没有 5.x 的 `is_causal` 后门**

A 版测试直接验了这一点：改最后一个 token，
行1（causal）前面的 logits **逐位不变**，A2D 模型前面的 logits 变化 0.92。

## 9.5 A 版实测

```
[hf-dllm-a2d-example] loss 5.7631 -> 5.2761   eval 4.9765
                        27,233 / 25,746 tok/s   rep-4 0.0000
C5  conversion: 0 missing, 0 unexpected, 0 altered
C2  vs hand-written loss: rel_diff 7.6e-08
```

慢 40% 的原因见 step 8.4：**是 `compute_loss` 的 python 开销，不是模型。**

## 9.6 解码：dllm 的 `steps` 是全序列预算

```python
# dllm/core/samplers/mdlm.py:131
num_blocks = math.ceil(max_new_tokens / block_size)   # 32/8 = 4
steps = math.ceil(steps / num_blocks)                  # 8/4 = 2   ← 每块 2 步
```

而 `notes/03` §7 的语义是「每块的迭代步数」，两者**同一个标签含义不同**：

| 配置 | dllm forward 次数 | A 版 sampler |
|---|---|---|
| gen 32, block 8, steps 8 | **8** | **32** |

A 版实测（同样 32 次 forward）：

```
手写 sampler   48.6 ms      MDLMSampler   101.0 ms      2.1x
```

**做速度对比时必须把预算换算到同一工作量**，否则测的是「谁配置的步数少」。

## 9.7 坑

| 坑 | 症状 |
|---|---|
| patch 没打 | `import dllm.pipelines.a2d` 直接炸 |
| 抄 dllm 的 loss | 测的是抄写，不是上游 |
| 直接构造 Trainer | 被 dataset / tokenizer 要求卡住 |
| 拿 `steps=8` 比解码速度 | 一个跑 8 步一个跑 32 步 |
| 忽略 `time_epsilon` | C2 对不上，且小 t 权重不同 |

## 9.8 验收命令

```sh
.venv/bin/python example/tests/test_dllm_a2d.py
.venv/bin/python example/check_ab.py all
.venv/bin/python example/check_ab.py decode-bench
```