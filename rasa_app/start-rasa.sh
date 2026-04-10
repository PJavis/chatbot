#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
DATA_SOURCE="${RASA_DATA_SOURCE:-/data/processed}"
DOCS_DIR="${RASA_KB_DIR:-$SCRIPT_DIR/docs}"
PORT="${RASA_PORT:-5005}"
RASA_DEBUG="${RASA_DEBUG:-0}"

if [ ! -f "$DATA_SOURCE/corpus.json" ]; then
  echo "Missing corpus.json at $DATA_SOURCE/corpus.json"
  echo "Run: python -m data.download"
  exit 1
fi

python "$SCRIPT_DIR/scripts/build_knowledge_docs.py" \
  --source "$DATA_SOURCE" \
  --target "$DOCS_DIR" \
  --workers 1

cd "$SCRIPT_DIR"
rasa train
LATEST_MODEL="$(ls -t "$SCRIPT_DIR"/models/*.tar.gz | head -n 1)"
python "$SCRIPT_DIR/scripts/patch_model_flow_retrieval_index.py" \
  --model "$LATEST_MODEL" \
  --cache-dir "$SCRIPT_DIR/.rasa/cache"
if [ "$RASA_DEBUG" = "1" ]; then
  exec rasa run --model "$LATEST_MODEL" --enable-api --port "$PORT" --cors "*" -vv
fi

exec rasa run --model "$LATEST_MODEL" --enable-api --port "$PORT" --cors "*" --quiet
