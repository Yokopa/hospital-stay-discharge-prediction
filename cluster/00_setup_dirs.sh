#!/bin/bash

# Sets up the project folder structure by creating all required directories
# relative to the script location for data, models, reports, results, and configs.

# Exit immediately if a command exits with a non-zero status
set -e

echo "Setting up project directories..."

# Get the base directory (parent of the directory where this script is located)
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Define the paths
DATA_DIR="$BASE_DIR/data"
RAW_DIR="$DATA_DIR/raw"
INTERIM_DIR="$DATA_DIR/interim"
PROCESSED_DIR="$DATA_DIR/processed"
MODELS_DIR="$BASE_DIR/models"
REPORTS_DIR="$BASE_DIR/reports"
FIGURES_DIR="$REPORTS_DIR/figures"
TABLES_DIR="$REPORTS_DIR/tables"
LOGS_DIR="$BASE_DIR/logs"
RESULTS_DIR="$BASE_DIR/results"
CONFIG_DIR="$BASE_DIR/config_experiment"

# Create directories
mkdir -p "$RAW_DIR" \
         "$INTERIM_DIR" \
         "$PROCESSED_DIR" \
         "$MODELS_DIR" \
         "$FIGURES_DIR" \
         "$TABLES_DIR" \
         "$LOGS_DIR" \
         "$RESULTS_DIR" \
         "$CONFIG_DIR"

echo "All required directories created successfully."
