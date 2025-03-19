#!/bin/bash
#SBATCH --account=def-rgreiner
#SBATCH --job-name=pretrain
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=64G
#SBATCH --output=job_output/%x/%j/out.txt
#SBATCH --error=job_output/%x/%j/err.txt

# Print starting directory and environment info
echo "Starting job at $(date)"
echo "Working directory: $(pwd)"
echo "Hostname: $(hostname)"

cd "$(pwd)"
module load StdEnv/2023
module load rdkit
module load python/3.11
# module load cuda
# module load cudacore/.11.0.2
# module load cudnn/8.0.3

# Activate virtual environment
source ~/venvs/gifflar/bin/activate
which python
python --version
pip list

# Add cwd directory to PYTHONPATH
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Run your application
python -m gifflar.train configs/pretraining/pretrain.yaml > log_pretrain.txt

# Print completion message
echo "Job completed at $(date)"