#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v uv >/dev/null; then
    python3 -m pip install --target .tools uv
    UV="$PWD/.tools/bin/uv"
else
    UV="$(command -v uv)"
fi
if [ ! -d vendor/i2rt ]; then
    git clone https://github.com/i2rt-robotics/i2rt.git vendor/i2rt
    git -C vendor/i2rt checkout "$(cat SDK_REVISION)"
fi
test "$(git -C vendor/i2rt rev-parse HEAD)" = "$(cat SDK_REVISION)"
# Keep the pinned SDK's opt-in integral controller reproducible; never overwrite local edits.
if git -C vendor/i2rt apply --reverse --check ../../patches/i2rt-integral.patch 2>/dev/null; then
    : # Already applied.
else
    git -C vendor/i2rt apply --check ../../patches/i2rt-integral.patch
    git -C vendor/i2rt apply ../../patches/i2rt-integral.patch
fi
"$UV" venv --python 3.11 --allow-existing .venv
"$UV" pip install --python .venv/bin/python -r requirements.lock --build-constraint scripts/build-constraints.txt
"$UV" pip install --python .venv/bin/python --no-deps -e .
