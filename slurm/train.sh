#!/bin/bash
#SBATCH --job-name=spoter-train
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16G
#SBATCH --qos=cs
#SBATCH --output=slurm/%j-train.out

# Train SPOTER on WLASL100 with epoch-specific checkpoints.
#
# Submit from the analysis/ directory:
#   sbatch slurm/train.sh
#
# The WLASL100 data CSVs must already be downloaded to spoter/data/.
# See slurm/download_data.sh if you haven't done that yet.

set -euo pipefail

module load python/3.12

ANALYSIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ANALYSIS_DIR}/../spoter/data"

source "${ANALYSIS_DIR}/.venv/bin/activate"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Data:   ${DATA_DIR}"
echo ""

python "${ANALYSIS_DIR}/scripts/train_spoter.py" \
    --training_set_path   "${DATA_DIR}/WLASL100_train_25fps.csv" \
    --validation_set_path "${DATA_DIR}/WLASL100_val_25fps.csv" \
    --testing_set_path    "${DATA_DIR}/WLASL100_test_25fps.csv" \
    --num_classes   100 \
    --hidden_dim    108 \
    --epochs        350 \
    --lr            0.001 \
    --seed          379 \
    --experiment_name wlasl100_spoter \
    "$@"
