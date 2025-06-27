#!/bin/bash

#SBATCH --job-name=baseline_los_all_models
#SBATCH --output=/data/users/ascarpellini/hospital_stay_discharge_prediction/logs/01_baseline_los_output_%A.o
#SBATCH --error=/data/users/ascarpellini/hospital_stay_discharge_prediction/logs/01_baseline_los_error_%A.e
#SBATCH --time=12:00:00
#SBATCH --partition=pibu_el8
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=anna.scarpellinipancrazi@students.unibe.ch
#SBATCH --mail-type=end

set -e
trap 'echo "Error occurred. Exiting script."; exit 1' ERR

echo "Job started at: $(date)"
start_time=$(date +%s)

WORKDIR="/data/users/ascarpellini/hospital_stay_discharge_prediction"
LOGDIR="$WORKDIR/results/baseline_los/logs"
mkdir -p "$LOGDIR"

# Load conda and activate env
module load Anaconda3/2022.05
eval "$(conda shell.bash hook)"
conda activate hospital-stay-discharge-prediction

cd "$WORKDIR/src"

# Define all datasets, param sets, and models
DATASETS=(
    nan
    nan_missing_flags_only
    nan_icd_blocks_only
    nan_icd_categories_only
    nan_age_bin
    nan_new_features
)

PARAM_SETS=(default default_balanced)

MODELS=(xgboost lightgbm catboost)

for DATASET in "${DATASETS[@]}"; do
    for PARAM_SET in "${PARAM_SETS[@]}"; do
        for MODEL in "${MODELS[@]}"; do
            LOGFILE="$LOGDIR/baseline_los_${DATASET}_${MODEL}_${PARAM_SET}.log"
            echo "Running: $DATASET | $MODEL | $PARAM_SET"
            python main.py \
                --step baseline_los \
                --dataset_name "$DATASET" \
                --model_name "$MODEL" \
                --param_set "$PARAM_SET" \
                > "$LOGFILE" 2>&1
        done
    done
done

echo "All baseline LOS runs complete. Logs saved in $LOGDIR"

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
