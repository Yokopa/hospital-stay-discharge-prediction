#!/bin/bash

#SBATCH --job-name=clean_data
#SBATCH --output=/data/users/ascarpellini/hospital_stay_discharge_prediction/logs/01_clean_data_output_%A.o
#SBATCH --error=/data/users/ascarpellini/hospital_stay_discharge_prediction/logs/01_clean_data_error_%A.e
#SBATCH --time=01:00:00
#SBATCH --partition=pibu_el8
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=anna.scarpellinipancrazi@students.unibe.ch
#SBATCH --mail-type=end

######################################################################
# Script: clean_raw_data.sh
#
# Description:
#   SLURM batch script to clean and merge raw laboratory and clinical data.
#   This script prepares the initial dataset used for all downstream
#   modeling tasks (e.g., LOS and discharge type prediction).
#
# Usage:
#   This script runs:
#     python main.py --step clean_only
#
#   The cleaned dataset will be saved to the default path specified in
#   config.CLEANED_MERGED_DATA_PATH.
#
# Environment Setup (one-time setup):
# -----------------------------------
# To run this script, ensure the conda environment is properly created.
# These steps are one-time setup steps to create the required environment
# for running the script. 
#
# NOTE: Once the environment is created, you only need
# to submit the script with sbatch (the environment is activated inside it).
#
#   1. Start an interactive session:
#        srun --time=02:00:00 --mem=2G --ntasks=1 --cpus-per-task=2 \
#             --partition=pibu_el8 --pty bash
#
#   2. Load Conda module:
#        module load Anaconda3/2022.05
#
#   3. Initialize Conda:
#        eval "$(conda shell.bash hook)"
#
#   4. Create environment from YAML:
#        conda env create -f /data/users/ascarpellini/hospital_stay_discharge_prediction/environment.yml --verbose
#
#   5. Activate the environment:
#        conda activate hospital-stay-discharge-prediction
#
#   6. When done, deactivate it:
#        conda deactivate
#
######################################################################

set -e

trap 'echo "Error occurred. Exiting script."; exit 1' ERR

WORKDIR="/data/users/ascarpellini/hospital_stay_discharge_prediction"

echo "Job started at: $(date)"
start_time=$(date +%s)

# Setup directories (adjust path if needed)
bash $WORKDIR/bash_scripts/00_setup_dirs.sh

# Load conda and activate environment
module load Anaconda3/2022.05
eval "$(conda shell.bash hook)"
conda activate hospital-stay-discharge-prediction

cd "$WORKDIR/src"

# Run cleaning step (adjust arguments accordingly)
python main.py --step clean_only
echo "Cleaned data should now be saved to: $WORKDIR/data/interim/cleaned_merged_data.csv"

end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Total duration: $((duration / 60)) minutes and $((duration % 60)) seconds"

conda deactivate