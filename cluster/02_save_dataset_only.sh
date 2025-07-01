#!/bin/bash

#SBATCH --job-name=prepare_dataset
#SBATCH --output=/data/users/ascarpellini/hospital_stay_discharge_prediction/logs/02_DT_ST_save_dataset_only_output_%A.o
#SBATCH --error=/data/users/ascarpellini/hospital_stay_discharge_prediction/logs/02_DT_BL_save_dataset_only_error_%A.e
#SBATCH --time=2-00:00:00
#SBATCH --partition=pibu_el8
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=anna.scarpellinipancrazi@students.unibe.ch
#SBATCH --mail-type=end

######################################################################
# Script: 02_save_dataset_only.sh
#
# Description:
#   General-purpose SLURM batch script to preprocess and save a dataset
#   for any prediction task (`los` or `discharge_type`) using any dataset
#   configuration defined in datasets.yaml.
#
#   This script runs:
#     python main.py --step save_dataset_only \
#                    --target <target_name> \
#                    --dataset_name <dataset_config> \
#                    --save_dataset
#
#   Example:
#     python main.py --step save_dataset_only \
#                    --target los \
#                    --dataset_name imputed_encoded_scaled_icd_blocks \
#                    --save_dataset
#
# Notes:
#   - Output logs are stored in logs/.
#   - Processed datasets are saved in data/processed/.
#   - To reuse this script, just modify the --target and --dataset_name.
#
######################################################################

set -e

trap 'echo "Error occurred. Exiting script."; exit 1' ERR

WORKDIR="/data/users/ascarpellini/hospital_stay_discharge_prediction"

echo "Job started at: $(date)"
start_time=$(date +%s)

# Load conda and activate environment
module load Anaconda3/2022.05
eval "$(conda shell.bash hook)"
conda activate hospital-stay-discharge-prediction

cd "$WORKDIR/src"

python main.py --step save_dataset_only --target discharge_type --dataset_name imputed_encoded_scaled_icd_blocks --save_dataset
echo "Cleaned data should now be saved to: $WORKDIR/data/interim/cleaned_merged_data.csv"

end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Total duration: $((duration / 60)) minutes and $((duration % 60)) seconds"

# Sleep briefly to ensure sacct has finished logging the job
sleep 10

# Print resource usage (or save to file)
echo "Resource usage:"
sacct -j "$SLURM_JOB_ID" --format=JobID,JobName,MaxRSS,MaxVMSize,Elapsed,CPUTime,State

conda deactivate
echo "Done"