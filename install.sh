#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$HOME/.local/bin" "$HOME/.codex/skills" "$HOME/.config/dream" "$HOME/.config/systemd/user"
ln -sfn "$REPO_DIR/dream/dream.py" "$HOME/.local/bin/dream"
ln -sfn "$REPO_DIR/dream/metrics.py" "$HOME/.local/bin/dream-metrics"
ln -sfn "$REPO_DIR/dream/stage_runner.py" "$HOME/.local/bin/dream-stage"
ln -sfn "$REPO_DIR/skill" "$HOME/.codex/skills/dream"
install -m 0644 "$REPO_DIR/dream/systemd/dream.service" "$HOME/.config/systemd/user/dream.service"
install -m 0644 "$REPO_DIR/dream/systemd/dream.timer" "$HOME/.config/systemd/user/dream.timer"
install -m 0644 "$REPO_DIR/dream/systemd/dream-exporter.service" "$HOME/.config/systemd/user/dream-exporter.service"

if [ ! -f "$HOME/.config/dream/config.toml" ]; then
    install -m 0644 "$REPO_DIR/dream/default-config.toml" "$HOME/.config/dream/config.toml"
fi

echo "dream CLI -> $HOME/.local/bin/dream"
echo "dream metrics -> $HOME/.local/bin/dream-metrics"
echo "dream stage runner -> $HOME/.local/bin/dream-stage"
echo "dream skill -> $HOME/.codex/skills/dream"
echo "dream config -> $HOME/.config/dream/config.toml"
echo "systemd units -> $HOME/.config/systemd/user/ (run systemctl --user daemon-reload separately)"
