#!/bin/bash

#SBATCH --job-name=discharge_type_log_reg
#SBATCH --output=/data/users/ascarpellini/hospital_stay_discharge_prediction/logs/03_DT_logreg_run_output_%A.o
#SBATCH --error=/data/users/ascarpellini/hospital_stay_discharge_prediction/logs/03_DT_logreg_run_error_%A.e
#SBATCH --time=06:00:00
#SBATCH --partition=pibu_el8
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-1
#SBATCH --mail-user=anna.scarpellinipancrazi@students.unibe.ch
#SBATCH --mail-type=end

######################################################################
# Script: 03_run_logreg_discharge.sh
#
# Description:
#   This SLURM batch script runs logistic regression for discharge type
#   classification on two dataset variants using SLURM array jobs.
#
#   Array Task IDs:
#     0 → imputed_encoded_scaled_icd_blocks
#     1 → imputed_encoded_scaled_icd_categories
#
#   Each task runs:
#     python main.py --step run_model --target discharge_type ...
#
# Requirements:
#   - Conda environment `hospital-stay-discharge-prediction` must exist
#   - `main.py` must be inside `src/`
#
# Outputs:
#   - Results are saved to results/discharge_type/*.csv
#   - Logs are written to logs/ and results/discharge_type/logs/
#
######################################################################

set -e
trap 'echo "Error occurred. Exiting script."; exit 1' ERR

echo "Job started at: $(date)"
start_time=$(date +%s)

WORKDIR="/data/users/ascarpellini/hospital_stay_discharge_prediction"
LOGDIR="$WORKDIR/results/discharge_type/logs"
mkdir -p "$LOGDIR"

# -------- Dataset names corresponding to SLURM array task IDs --------
DATASETS=("imputed_encoded_scaled_icd_blocks" "imputed_encoded_scaled_icd_categories")
# Select dataset name based on SLURM_ARRAY_TASK_ID
DATASET_NAME=${DATASETS[$SLURM_ARRAY_TASK_ID]}

module load Anaconda3/2022.05
eval "$(conda shell.bash hook)"
conda activate hospital-stay-discharge-prediction

cd "$WORKDIR/src"

echo "Running logistic regression on dataset: $DATASET_NAME"

# Run main script with args
python main.py \
  --step run_model \
  --target discharge_type \
  --dataset_name "$DATASET_NAME" \
  --model_name logistic_regression \
  --param_set default

echo "All discharge type runs complete. Logs saved in $LOGDIR"

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

