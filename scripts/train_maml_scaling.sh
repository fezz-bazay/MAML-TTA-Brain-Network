#!/bin/bash
#SBATCH --job-name=maml_scaling
#SBATCH --account=project_2019360
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=./logs/scaling/maml_scaling_%j.out
#SBATCH --error=./logs/scaling/maml_scaling_%j.err

module --force purge
module load pytorch/2.0
cd "$(dirname "$0")" || exit 1
# Activate your environment (update path if needed)
source ./env/bin/activate
export PYTHONPATH=".:$PYTHONPATH"
export WANDB_MODE=disabled
mkdir -p ./logs/scaling

echo "MAML-TTA Scaling N=4 — Start: $(date)"
python source/experiments/scaling/train_maml_scaling.py --n_sites=4
EXIT_CODE=$?
echo "End: $(date) | Exit: ${EXIT_CODE}"
exit ${EXIT_CODE}