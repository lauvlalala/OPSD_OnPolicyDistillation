#!/bin/bash
# Pull SWE-Gym Lite Docker images (230 total, ~2-6GB each)
#
# Usage:
#   bash scripts/opd/pull_swe_images.sh           # pull all
#   bash scripts/opd/pull_swe_images.sh 5         # pull first 5 (smoke test)
#   bash scripts/opd/pull_swe_images.sh 0 10      # pull index 0-9
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
IMAGE_LIST="${REPO_ROOT}/data/swe_gym/image_list.txt"

if [ ! -f "$IMAGE_LIST" ]; then
    echo "Error: $IMAGE_LIST not found. Run prepare_swe_gym.py first."
    exit 1
fi

TOTAL=$(wc -l < "$IMAGE_LIST")
START=${1:-0}
COUNT=${2:-$TOTAL}
END=$((START + COUNT))
if [ $END -gt $TOTAL ]; then END=$TOTAL; fi

echo "=== Pulling SWE-Gym images [$START, $END) of $TOTAL ==="

PULLED=0
FAILED=0
i=0
while IFS= read -r image; do
    if [ $i -lt $START ]; then i=$((i+1)); continue; fi
    if [ $i -ge $END ]; then break; fi

    echo "[$((i+1))/$END] Pulling $image ..."
    if docker pull "$image"; then
        PULLED=$((PULLED+1))
    else
        echo "  FAILED: $image"
        FAILED=$((FAILED+1))
    fi
    echo ""
    i=$((i+1))
done < "$IMAGE_LIST"

echo "=== Done: pulled $PULLED, failed $FAILED ==="
