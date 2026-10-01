#!/usr/bin/env bash
# 创建指向 /mnt/extra 的软链接。
#
# 为什么用软链接而不是把大文件留在项目目录:
#   - 项目在 btrfs (/home), 大文件 (数据/权重) 放这里既占空间又拖慢 btrfs
#   - /mnt/extra 是 ext4, 专门给大文件
#   - 这些路径全部在 .gitignore 里, 所以软链接本身不进 git
#     → 换机器 / 换 workspace 时重跑一次这个脚本即可
#
# .gitignore 里必须写 `/data` 而不是 `/data/`。
# git 的 `/data/` 只匹配目录, 而软链接对 git 是个文件, 匹配不到 ——
# 结果软链接本身会被提交进仓库。无斜杠的写法同时覆盖目录和软链接。
#
# 幂等: 已存在的链接会重建, 不会报错。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTRA="${EXTRA_ROOT:-/mnt/extra}"

link() {
    local name="$1" target="$2"
    if [[ ! -e "$target" ]]; then
        echo "SKIP  $name -> $target   (目标不存在)"
        return
    fi
    if [[ -L "$ROOT/$name" ]]; then
        rm "$ROOT/$name"
    elif [[ -e "$ROOT/$name" ]]; then
        echo "ERROR $name 已存在且不是软链接, 手动处理: $ROOT/$name"
        return 1
    fi
    ln -s "$target" "$ROOT/$name"
    echo "OK    $name -> $target"
}

link data        "$EXTRA/data"
link checkpoints "$EXTRA/ckpt"

echo
echo "当前状态:"
ls -la "$ROOT" | grep -E '^l' || echo "  (没有软链接)"

# logs/ 不链接 —— 纯文本产物, 而且 logs/dev/ 要进 git
