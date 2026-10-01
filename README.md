# ARnDlm

**A**uto **R**egressive a*n*d **D**iffusion language model.

在**同一套 tokenizer / vocab / 数据集 / venv** 下，把 AR baseline 和 DLM 放到
可比的位置上，用实测数字回答「DLM 到底行不行」。

## 三条线

**「minimind-dlm」= 用 minimind-dlm 的思路（notes/03-minimind-dlm.md）做 A2D。**

| 线 | 做什么 | 在哪 |
|---|---|---|
| **T1** | 训 AR 基准模型 | T1a `src/transformer/` · T1b `src/minimind/` |
| **T2** | 对基准模型做 A2D | T2a `src/a2d/`（minimind-dlm）· T2b `src/dllm-a2d/` · T2c 对开源模型做 |
| **T3** | 直接用 `dllm` 包训 DLM | `src/dllm/` |

`src/base/`（公用工具）和 `src/utils/`（日志/web/设备）被所有线共用。
依赖方向和开工顺序见 `src/README.md`。步骤与验收见 `ROADMAP.md`。

## 地图

```
notes/      纯知识 —— 概念、原理、坑。换台机器依然成立
logs/dev/   手写开发记录 —— 本机实测、踩过的坑
logs/run/   程序产物 —— 训练日志、summary.json、曲线 PNG
example/    参考实现 —— 我写，你对照学
test/       验证脚本 —— 每个只回答「能不能 / 对不对」
src/        正式实现 —— 你写（base / utils / transformer / minimind /
            a2d / dllm-a2d / dllm，见 src/README.md）
```

规则见 `AGENTS.md`，命令见 `USAGE.md`。

## refs

```
本项目取代 ~/Projects/dlm（只读参考，不再改动）
dllm 包      ~/Public/dllm
minimind     ~/Public/minimind
minimind-dlm https://jingyaogong.github.io/space/dllm_intro
```
