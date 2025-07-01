#!/bin/bash

#SBATCH --job-name=xgboost_discharge_3
#SBATCH --output=/data/users/ascarpellini/hospital_stay_discharge_prediction/logs/05_dt_3_xgboost_output_%A_%a.o
#SBATCH --error=/data/users/ascarpellini/hospital_stay_discharge_prediction/logs/05_dt_3_xgboost_error_%A_%a.e
#SBATCH --time=04:00:00
#SBATCH --partition=pibu_el8
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-5
#SBATCH --mail-user=anna.scarpellinipancrazi@students.unibe.ch
#SBATCH --mail-type=end

set -e
trap 'echo "Error occurred. Exiting script."; exit 1' ERR

echo "Starting Discharge Classification | $(date)"
start_time=$(date +%s)

WORKDIR="/data/users/ascarpellini/hospital_stay_discharge_prediction"
cd "$WORKDIR"

module load Anaconda3/2022.05
eval "$(conda shell.bash hook)"
conda activate hospital-stay-discharge-prediction

DATASETS=("nan" "nan_missing_flags_only" "nan_icd_blocks_only" "nan_icd_categories_only" "nan_age_bin" "nan_new_features")
DATASET_NAME=${DATASETS[$SLURM_ARRAY_TASK_ID]}

cd "$WORKDIR/src"
echo "Running discharge model on dataset: $DATASET_NAME"

python main.py \
  --step run_model \
  --target discharge_type \
  --model_name xgboost \
  --dataset_name "$DATASET_NAME" \
  --param_set trial_fast

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
