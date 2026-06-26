#!/bin/bash
#SBATCH --job-name=bnt_mldg
#SBATCH --account=project_2019360
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=./logs/baselines/bnt_mldg_%j.out
#SBATCH --error=./logs/baselines/bnt_mldg_%j.err

module --force purge
module load pytorch/2.0
cd "$(dirname "$0")" || exit 1
# Activate your environment (update path if needed)
source ./env/bin/activate
export PYTHONPATH=".:$PYTHONPATH"
export WANDB_MODE=disabled
mkdir -p ./logs/baselines

echo "BNT + MLDG — Start: $(date)"
python source/experiments/baselines/train_bnt_mldg.py
EXIT_CODE=$?
echo "End: $(date) | Exit: ${EXIT_CODE}"
exit ${EXIT_CODE}