#!/bin/bash
#SBATCH --job-name=spoter-analysis
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=8G
#SBATCH --qos=cs
#SBATCH --output=slurm/%j-analysis.out

# General-purpose analysis job: extraction, probing, alignment.
#
# Submit from the analysis/ directory, passing the script name as first arg:
#   sbatch slurm/analysis.sh extract_attention.py --checkpoint results/checkpoints/wlasl100_spoter_epoch350.pth
#   sbatch slurm/analysis.sh extract_mbert.py
#   sbatch slurm/analysis.sh run_probing.py

set -euo pipefail

module load python/3.12

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch slurm/analysis.sh <script_name.py> [args...]"
    exit 1
fi

ANALYSIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$1"
shift

source "${ANALYSIS_DIR}/.venv/bin/activate"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "Script: ${SCRIPT}"
echo ""

python "${ANALYSIS_DIR}/scripts/${SCRIPT}" "$@"
