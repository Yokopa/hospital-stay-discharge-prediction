#!/bin/bash

"""
Script: clean_raw_data.py
Purpose: Cleans laboratory and clinical raw data for downstream modeling.

Environment Setup:
------------------
To run this script, first create and activate the appropriate conda environment:

1. Create the environment (once):
    conda env create -f environment.yml

2. Activate it:
    conda activate hospital-stay-discharge-prediction

Include `module load anaconda` and `conda activate hospital-stay-discharge-prediction` in your job script.
"""

#SBATCH --job-name=/data/users/ascarpellini/hospital_stay_discharge_prediction/reports/logs/01_clean_data
#SBATCH --output=01_clean_raw_data.out
#SBATCH --error=01_clean_raw_data.err
#SBATCH --time=01:00:00
#SBATCH --partition=pibu_el8
#SBATCH --mem=4G
#SBATCH --mail-user=anna.scarpellinipancrazi@students.unibe.ch
#SBATCH --mail-type=end

WORKDIR="/data/users/ascarpellini/hospital_stay_discharge_prediction"

# Start timing
echo "Job started at: $(date)"
start_time=$(date +%s)

./setup_dirs.sh
module load anaconda
source activate hospital-stay-discharge-prediction

cd "$WORKDIR/src"
python data_preparation/clean_raw_data.py

# End timing
end_time=$(date +%s)
echo "Job ended at: $(date)"

# Compute duration
duration=$((end_time - start_time))
echo "Total duration: $((duration / 60)) minutes and $((duration % 60)) seconds"

