#!/bin/bash
#SBATCH --job-name=bnt_source_only
#SBATCH --account=project_2019360
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=./logs/bnt_source_only_%j.out
#SBATCH --error=./logs/bnt_source_only_%j.err

module --force purge
module load pytorch/2.0
cd "$(dirname "$0")" || exit 1
# Activate your environment (update path if needed)
source ./env/bin/activate
export PYTHONPATH=".:$PYTHONPATH"
export WANDB_MODE=disabled
mkdir -p ./logs

echo "BNT SOURCE-ONLY — Start: $(date)"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import source; print('Module source OK')" || exit 1
python source/experiments/baselines/train_bnt_source_only.py
EXIT_CODE=$?
echo "End: $(date) | Exit: ${EXIT_CODE}"
exit ${EXIT_CODE}