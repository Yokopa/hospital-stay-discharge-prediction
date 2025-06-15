#!/bin/bash

# Base directory (adjust to your actual working directory on cluster)
BASE_DIR="/data/users/ascarpellini/hospital_stay_discharge_prediction"

# List of directories to create
DIRS=(
    "$BASE_DIR/data/raw"
    "$BASE_DIR/data/interim"
    "$BASE_DIR/models"
    "$BASE_DIR/reports/figures"
    "$BASE_DIR/reports/tables"
    "$BASE_DIR/reports/logs"
)

# Create each directory if it doesn't exist
for dir in "${DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "Created directory: $dir"
    else
        echo "Directory already exists: $dir"
    fi
done
