#!/usr/bin/env bash
# .githooks/install.sh — One-time activation of repo-tracked git hooks.
#
# Why: a tracked hook file (`.githooks/pre-commit`) does nothing until git
# is told to look in `.githooks/` instead of the per-clone `.git/hooks/`.
# This script sets that config in the local clone (no commit needed).
#
# What: runs `git config core.hooksPath .githooks` for the current clone
# and reports the current hook source so you can verify it took effect.
#
# How: run once after `git clone`. Idempotent — safe to re-run.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

echo "[install-hooks] setting core.hooksPath = .githooks ..."
git config core.hooksPath .githooks

# Make every hook in .githooks/ executable.
echo "[install-hooks] chmod +x .githooks/*"
chmod +x .githooks/*

echo
echo "[install-hooks] done. Active hook source:"
echo "    $(git config --get core.hooksPath)"
echo
echo "[install-hooks] Available hooks:"
for h in .githooks/*; do
    [[ -f "$h" && -x "$h" && "$(basename "$h")" != "install.sh" ]] || continue
    printf '    %s\n' "$(basename "$h")"
done
echo
echo "[install-hooks] To bypass a hook for one commit (use sparingly):"
echo "    git commit --no-verify -m '...'"
