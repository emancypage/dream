#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$HOME/.local/bin" "$HOME/.codex/skills" "$HOME/.config/dream"
ln -sfn "$REPO_DIR/dream/dream.py" "$HOME/.local/bin/dream"
ln -sfn "$REPO_DIR/skill" "$HOME/.codex/skills/dream"

if [ ! -f "$HOME/.config/dream/config.toml" ]; then
    install -m 0644 "$REPO_DIR/dream/default-config.toml" "$HOME/.config/dream/config.toml"
fi

echo "dream CLI -> $HOME/.local/bin/dream"
echo "dream skill -> $HOME/.codex/skills/dream"
echo "dream config -> $HOME/.config/dream/config.toml"
