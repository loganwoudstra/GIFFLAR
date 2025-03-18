#!/bin/bash
#SBATCH --account=def-rgreiner
#SBATCH --job-name=data-aquisition
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=job_output/%x/%j/out
#SBATCH --error=job_output/%x/%j/err

# Print starting directory and environment info
echo "Starting job at $(date)"
echo "Working directory: $(pwd)"
echo "Hostname: $(hostname)"

cd "$(pwd)"
module load StdEnv/2023
module load rdkit
module load python/3.11

# Activate virtual environment
source ~/venvs/gifflar/bin/activate
which python
python --version
pip list

# Add cwd directory to PYTHONPATH
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Run your application
python gifflar/acquisition/collect_smiles_data.py

# Print completion message
echo "Job completed at $(date)"