#!/bin/bash
# Download preprocessed WLASL100 skeleton CSV files from the SPOTER GitHub
# release: https://github.com/matyasbohacek/spoter/releases/tag/supplementary-data
#
# Run this from the LOGIN NODE (not via SLURM — compute nodes lack internet).
#
#   bash scripts/download_data.sh
#
# BEFORE RUNNING: verify the exact filenames by visiting the release page above.
# Update the FILENAMES array below if they differ from what's listed here.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPOTER_DIR="${SCRIPT_DIR}/../../spoter"
DATA_DIR="${SPOTER_DIR}/data"
RELEASE_BASE="https://github.com/matyasbohacek/spoter/releases/download/supplementary-data"

mkdir -p "${DATA_DIR}"

# Verify these against the GitHub release page before running.
FILENAMES=(
    "WLASL100_train_25fps.csv"
    "WLASL100_val_25fps.csv"
    "WLASL100_test_25fps.csv"
)

echo "Downloading WLASL100 data to ${DATA_DIR}"
echo ""

for FNAME in "${FILENAMES[@]}"; do
    DEST="${DATA_DIR}/${FNAME}"
    if [[ -f "${DEST}" ]]; then
        echo "Already exists, skipping: ${FNAME}"
        continue
    fi
    URL="${RELEASE_BASE}/${FNAME}"
    echo "Downloading: ${URL}"
    wget -q --show-progress -O "${DEST}" "${URL}" || {
        echo ""
        echo "ERROR: Could not download ${URL}"
        echo "Check https://github.com/matyasbohacek/spoter/releases/tag/supplementary-data"
        echo "for the correct filenames and update the FILENAMES array in this script."
        rm -f "${DEST}"
        exit 1
    }
done

echo ""
echo "Done. Files in ${DATA_DIR}:"
ls -lh "${DATA_DIR}"
