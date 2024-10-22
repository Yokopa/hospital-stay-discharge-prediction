# Predictive Modeling of Hospital Length of Stay and Discharge Type

## 1. Overview
This project aims to predict hospital length of stay (LOS) and discharge type using initial lab values, patient demographic data, and primary diagnosis information. The predictive models will help provide actionable insights to clinical staff and improve hospital resource management.

## 2. Structure
The repository is organized as follows:
- data/: Contains the raw and cleaned datasets used for the analysis.
- R/: Folder containing R scripts for data processing, modeling, and validation.
- notebooks/: RMarkdown files detailing each step of the analysis, from data exploration to modeling.
- reports/: Contains final reports, plots, and figures summarizing the findings.

## 3. Datasets
- **Lab Data**: Sourced from a CSV file (RITM0154633_lab.csv), which ontains patient lab results.
- **Clinical Data**: Sourced from a CSV file (RITM0154633_main.csv), which includes patient demographics (age, sex), clinical details (discharge type, length of stay), and primary diagnosis information.

## 4. Methodology
The following steps were taken in the analysis:

### 4.1 Data Exploration
- Description of exploratory analysis performed, including visualization and statistical summaries of key variables such as LOS, discharge type, and lab results.

### 4.2 Data Cleaning
- Steps taken to clean and preprocess the datasets (e.g., handling missing values, removing duplicates, dealing with erroneous entries).

### 4.3 Data Merging
- Describe how the clinical and lab datasets were merged and the challenges encountered in this process.

### 4.4 Modeling Approaches
- Predicting Length of Stay: Explain the modeling techniques used (e.g., linear regression, machine learning methods).
- Predicting Discharge Type: Detail the classification methods used (e.g., logistic regression, decision trees).

### 4.5 Model Validation
- Cross-validation techniques, performance metrics (accuracy, sensitivity, specificity, AUC, etc.), and any statistical tests performed to evaluate the models.

## 5. Results

## 6. Future works

## 7. Reference papers
- “A Comparative Study of Pattern Recognition Algorithms for Predicting the Inpatient Mortality Risk Using Routine Laboratory Measurements” (https://doi.org/10.1007/s10462-018-9625-3)
- “Machine Learning Prediction of Hypoglycemia and Hyperglycemia From Electronic Health Records” (10.2196/36176)

## 8. How to run this project
Prerequisites:

    Specify the software environment (e.g., R, RStudio), and list required R packages (e.g., tidyverse, caret, randomForest, etc.).

Steps to Run:

    Clone the repository:

    bash

git clone https://github.com/yourusername/predictive-modeling-hospital-stay.git

Install required packages:

bash

install.packages(c('tidyverse', 'caret', 'glmnet', 'randomForest'))

Run scripts:

    Run data_cleaning.R to clean the datasets.
    Run modeling.R to train and validate the models.
    For detailed step-by-step procedures, see the notebooks/ folder.
