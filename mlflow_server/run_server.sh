#!/usr/bin/env sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"
PYTHON=${PYTHON:-"$ROOT/.venv/bin/python"}
if [ ! -x "$PYTHON" ] && [ -x "$ROOT/.venv/Scripts/python.exe" ]; then
  PYTHON="$ROOT/.venv/Scripts/python.exe"
fi
exec "$PYTHON" -B "$ROOT/mlflow_server/server.py" --mode s3
