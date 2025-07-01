#!/bin/bash

#SBATCH --job-name=whole_los_regression
#SBATCH --output=/data/users/ascarpellini/hospital_stay_discharge_prediction/logs/06_whole_los_reg_%A_%a.o
#SBATCH --error=/data/users/ascarpellini/hospital_stay_discharge_prediction/logs/06_whole_los_reg_%A_%a.e
#SBATCH --time=02:00:00
#SBATCH --partition=pibu_el8
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-17   # 6 datasets * 3 models = 18 tasks
#SBATCH --mail-user=anna.scarpellinipancrazi@students.unibe.ch
#SBATCH --mail-type=end

set -e
trap 'echo "Error occurred. Exiting script."; exit 1' ERR

echo "Starting LOS Regression | $(date)"
start_time=$(date +%s)

WORKDIR="/data/users/ascarpellini/hospital_stay_discharge_prediction"
cd "$WORKDIR"

module load Anaconda3/2022.05
eval "$(conda shell.bash hook)"
conda activate hospital-stay-discharge-prediction

# Define datasets and models
DATASETS=( "nan_icd_blocks_only" "nan_icd_categories_only" "nan_new_features")
# MODELS=("xgboost") "lightgbm" "catboost" 

# Calculate indices
NUM_DATASETS=${#DATASETS[@]}
NUM_MODELS=${#MODELS[@]}

# Get dataset and model indices based on SLURM_ARRAY_TASK_ID
DATASET_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_MODELS ))
MODEL_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_MODELS ))

DATASET_NAME=${DATASETS[$DATASET_IDX]}
MODEL_NAME=${MODELS[$MODEL_IDX]}

cd "$WORKDIR/src"
echo "Running LOS regression on dataset: $DATASET_NAME with model: $MODEL_NAME"

python main.py \
  --step run_model \
  --target los \
  --mode regression \
  --model_name "$MODEL_NAME" \
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
