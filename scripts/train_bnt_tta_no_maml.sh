#!/bin/bash
#SBATCH --job-name=tta_no_maml
#SBATCH --account=project_2019360
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=./logs/ablation/tta_no_maml_%j.out
#SBATCH --error=./logs/ablation/tta_no_maml_%j.err

module --force purge
module load pytorch/2.0
cd "$(dirname "$0")" || exit 1
# Activate your environment (update path if needed)
source ./env/bin/activate
export PYTHONPATH=".:$PYTHONPATH"
export WANDB_MODE=disabled
mkdir -p ./logs/ablation

echo "BNT-TTA (w/o MAML) k=3 — Start: $(date)"
python source/experiments/ablation/train_bnt_tta_no_maml.py
EXIT_CODE=$?
echo "End: $(date) | Exit: ${EXIT_CODE}"
exit ${EXIT_CODE}