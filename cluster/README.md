This folder contains bash scripts to run the analysis on the cluster.
The scripts are numbered to indicate the order in which they have been executed.

> Note: The Python scripts for the analysis are located in the `src/` folder.

---

## Environment Setup

To ensure reproducibility, use the provided `environment.yml` file to create a conda environment.

### Interactive Setup (Recommended for Development — only needed once!)

Start an interactive session on the IBU cluster to install dependencies and test things manually:

```bash
# Start an interactive session
srun --partition=standard --mem=8G --time=01:00:00 --pty bash

# Load Anaconda
module load anaconda

# Create the environment from the YAML file (only needed once)
conda env create -f environment.yml

# Activate the environment
conda activate hospital-stay-discharge-prediction
```
Once the environment is created, you can activate it directly in your bash scripts using:

```bash
# Load Anaconda
module load anaconda

# Activate your environment
source activate hospital-stay-discharge-prediction
```