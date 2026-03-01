#!/bin/bash
#SBATCH --job-name=spoter-tests
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --qos=cs
#SBATCH --output=slurm/%j-tests.out

# Run all pytest sanity-check tests (CPU only, no GPU needed).
#
# Submit from the analysis/ directory:
#   sbatch slurm/tests.sh

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
module load python/3.12

ANALYSIS_DIR="${SLURM_SUBMIT_DIR}"

source "${ANALYSIS_DIR}/.venv/bin/activate"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo ""

cd "${ANALYSIS_DIR}"
pytest tests/ -v
