#!/bin/bash
#SBATCH --job-name=maml_bnt_tta
#SBATCH --account=project_2019360
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=./logs/maml_bnt_tta_%j.out
#SBATCH --error=./logs/maml_bnt_tta_%j.err

module --force purge
module load pytorch/2.0
cd "$(dirname "$0")" || exit 1
# Activate your environment (update path if needed)
source ./env/bin/activate
export PYTHONPATH=".:$PYTHONPATH"
export WANDB_MODE=disabled
mkdir -p ./logs

echo "BNT-MAML-TTA k=3 — Start: $(date)"
python source/experiments/our_method/train_bnt_maml_tta.py --inner_lr=0.001 --k_shot=3 --num_inner_steps=3
EXIT_CODE=$?
echo "End: $(date) | Exit: ${EXIT_CODE}"
exit ${EXIT_CODE}