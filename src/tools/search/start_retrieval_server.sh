#!/bin/bash
# Start the local dense retrieval server for SearchQA training.
#
# Prerequisites:
#   pip install faiss-gpu transformers uvicorn fastapi
#   Download corpus + index (see below)
#
# Usage:
#   bash src/tools/search/start_retrieval_server.sh
#
# Download corpus and build index:
#   python -c "
#   from huggingface_hub import snapshot_download
#   snapshot_download('PeterJinGo/search-r1-retrieval-data', local_dir='data/retrieval')
#   "

set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Adjust paths to your setup
CORPUS_PATH=${CORPUS_PATH:-"${REPO_ROOT}/data/retrieval/corpus.jsonl"}
INDEX_PATH=${INDEX_PATH:-"${REPO_ROOT}/data/retrieval/e5_Flat.index"}
MODEL_PATH=${RETRIEVAL_MODEL:-"intfloat/e5-base-v2"}
PORT=${PORT:-8000}

echo "Starting retrieval server on port ${PORT}..."
echo "  Corpus: ${CORPUS_PATH}"
echo "  Index: ${INDEX_PATH}"
echo "  Model: ${MODEL_PATH}"

# Use the retrieval server from SDAR/verl examples if available,
# otherwise point to a local copy
RETRIEVAL_SERVER="${REPO_ROOT}/src/tools/search/retrieval_server.py"

if [ ! -f "$RETRIEVAL_SERVER" ]; then
    echo "retrieval_server.py not found at $RETRIEVAL_SERVER"
    echo "Please copy from verl examples or SDAR repo."
    exit 1
fi

python3 "$RETRIEVAL_SERVER" \
    --corpus_path "$CORPUS_PATH" \
    --index_path "$INDEX_PATH" \
    --model_path "$MODEL_PATH" \
    --port $PORT \
    --pooling_method mean \
    --max_length 256
