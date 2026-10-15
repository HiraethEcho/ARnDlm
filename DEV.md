# DEV 分支

在DEV 分支开发正式代码，以*-example为参考。

目录中要写正式代码的核心内容
```
.
├── a2d # 转化AR模型为DLM的代码
├── checkpoints -> /mnt/extra/ckpt # 保存模型
├── data -> /mnt/extra/data # 保存数据
├── models # 加载一个模型并推理的所需代码
│   ├── config.py
│   ├── model-trans.py # 用transformers包的代码
│   ├── model.py
│   ├── tier1.yaml
│   ├── tokenizer.json
│   └── tokenizer.py
├── prepare # 训练前做准备的代码
│   ├── data.py
│   ├── device.py
│   ├── prepare.py
│   └── seed.py
├── ROADMAP.md
├── scripts # 从这里执行py代码，包括环境初始化、数据准备、训练、使用模型等
├── test # 测试，应该移到 scripts 里
├── trainer # 运行训练的代码
└── USAGE.md
```
