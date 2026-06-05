#!/bin/bash
# Load SWE-Gym Docker images from shared NFS storage into local Docker daemon.
# Run this on any training machine before starting SWE-Gym training.
#
# Usage:
#   bash scripts/opd/load_swe_images.sh              # load all available
#   bash scripts/opd/load_swe_images.sh --check      # only show missing images
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
IMAGE_DIR="${REPO_ROOT}/data/swe_gym/docker_images"
IMAGE_LIST="${REPO_ROOT}/data/swe_gym/image_list.txt"

if [ ! -d "$IMAGE_DIR" ]; then
    echo "Error: $IMAGE_DIR not found."
    exit 1
fi

CHECK_ONLY=false
if [ "${1:-}" = "--check" ]; then CHECK_ONLY=true; fi

# Get list of locally available images
LOCAL_IMAGES=$(docker images --format '{{.Repository}}:{{.Tag}}' | sort)

LOADED=0
SKIPPED=0
MISSING=0

for tarfile in "$IMAGE_DIR"/*.tar.gz; do
    [ -f "$tarfile" ] || continue
    basename=$(basename "$tarfile" .tar.gz)
    full_name="xingyaoww/sweb.eval.x86_64.${basename}:latest"

    if echo "$LOCAL_IMAGES" | grep -q "^${full_name}$"; then
        SKIPPED=$((SKIPPED+1))
        continue
    fi

    if [ "$CHECK_ONLY" = true ]; then
        echo "MISSING: $full_name"
        MISSING=$((MISSING+1))
        continue
    fi

    echo "Loading $basename ..."
    docker load < "$tarfile" 2>&1 | tail -1
    LOADED=$((LOADED+1))
done

if [ "$CHECK_ONLY" = true ]; then
    echo "=== $MISSING missing, $SKIPPED already loaded ==="
else
    echo "=== Loaded $LOADED, skipped $SKIPPED (already present) ==="
fi
