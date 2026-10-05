"""common/config.py -- pyyaml 读 configs/<tier>.yaml，产出 frozen dataclass 和共享常量。

yaml 数值写成 3.0e-4 形式（YAML 1.1 浮点须有小数点）。未知键报错。
常量：PAD/BOS/EOS/MASK、VOCAB、BLOCK_SIZE、DTYPE 等。
参考实现：common/config-example.py
"""
