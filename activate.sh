#!/usr/bin/env bash

# Reuse the original openpi-umi dependencies and caches while importing code
# from this robustness worktree.
OLD_OPENPI=/data2/hzl_workspace_for_pi/openpi-umi
NEW_OPENPI=/data2/hzl_workspace_for_pi/openpi-umi-robustness

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Please source this script: source $NEW_OPENPI/activate.sh" >&2
    exit 1
fi

source "$OLD_OPENPI/.venv/bin/activate"
export PYTHONPATH="$NEW_OPENPI/src:$NEW_OPENPI/packages/openpi-client/src:$NEW_OPENPI/third_party/lerobot${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="$OLD_OPENPI/.cache"
cd "$NEW_OPENPI"

echo "Activated openpi-umi-robustness"
echo "Python: $(command -v python)"
echo "Workspace: $NEW_OPENPI"
