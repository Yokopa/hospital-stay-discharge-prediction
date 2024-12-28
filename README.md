# Predictive Modeling of Hospital Length of Stay and Discharge Type

## Introduction

Hospital length of stay (LOS) and discharge type are key metrics in healthcare, shaping patient outcomes and resource allocation. This project aims to lay the groundwork for predictive modeling of LOS and discharge type by analyzing clinical and laboratory data, including patient demographics, diagnoses, and outcomes.

Initial efforts focus on data cleaning and exploratory analysis to address missing values, duplicates, and outliers, ensuring high-quality datasets for modeling. 

## Repository Structure

The repository is organized as follows:
- scripts/: Contains R scripts for data cleaning and exploration.
  
## Methodology

### 1. Data Source and Overview

The dataset, provided by the Insel Data Science Center (IDSC) of Bern, contains clinical and laboratory data spanning approximately 16 years from Inselspital, the university hospital of Bern. The data includes:
- **Laboratory dataset**: Detailed test results from hospital visits.
- **Clinical dataset**: Patient demographics, LOS, discharge type, and diagnosis codes (ICD codes).

### 2. Data Inspection

- Translated column names (e.g., German to English).
- Checked dimensions, structure, summary statistics, missing values, and duplicates.
- Extracted unique test abbreviations and names.

### 3. Data Cleaning

#### Laboratory Dataset
- Imputed missing `test_abbr` values using `test_name`.
- Removed invalid numerical results and duplicates.
- Applied plausibility checks based on medical thresholds.

#### Clinical Dataset
- Removed invalid age values and focused on adult patients.
- Translated discharge type entries and addressed missing values.

### 4. Data Merging

- Merged datasets using shared patient and case identifiers.
- Applied Pareto filtering to exclude tests with >80% missing values.
- Aggregated test results (e.g., mean, median) for each patient and case.

### 5. Exploratory Data Analysis (EDA)

- Analyzed demographic statistics, LOS patterns, and diagnosis distributions.
- Visualized relationships between lab results, LOS, and clinical variables.
- Identified and handled outliers in LOS and lab results.

## Key Insights

- **Patient Demographics**: Age and gender distributions across LOS and discharge types.
- **Diagnosis Trends**: Frequency and categorization of principal diagnoses.
- **Lab Tests and LOS**: Correlation analysis between lab test results and hospital stay duration.

## Dependencies

The project uses R (version 4.4.1) and the following packages:
- Data manipulation: `data.table`, `dplyr`, `tidyr`
- Visualization: `ggplot2`, `ggcorrplot`, `reshape2`, `RColorBrewer`

Refer to the `DESCRIPTION` file for the full list of package dependencies.

## How to Use

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/los-prediction.git
   cd los-prediction

## Reference papers
- “A Comparative Study of Pattern Recognition Algorithms for Predicting the Inpatient Mortality Risk Using Routine Laboratory Measurements” (https://doi.org/10.1007/s10462-018-9625-3)
- “Machine Learning Prediction of Hypoglycemia and Hyperglycemia From Electronic Health Records” (doi:10.2196/36176)
