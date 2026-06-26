#!/bin/bash
#SBATCH --job-name=maml_no_tta
#SBATCH --account=project_2019360
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=./logs/ablation/maml_no_tta_%j.out
#SBATCH --error=./logs/ablation/maml_no_tta_%j.err

module --force purge
module load pytorch/2.0
cd "$(dirname "$0")" || exit 1
# Activate your environment (update path if needed)
source ./env/bin/activate
export PYTHONPATH=".:$PYTHONPATH"
export WANDB_MODE=disabled
mkdir -p ./logs/ablation

echo "ABLATION MAML sans TTA — Start: $(date)"
python source/experiments/ablation/train_bnt_maml_no_tta.py --inner_lr=0.001 --k_shot=3 --num_inner_steps=3
EXIT_CODE=$?
echo "End: $(date) | Exit: ${EXIT_CODE}"
exit ${EXIT_CODE}