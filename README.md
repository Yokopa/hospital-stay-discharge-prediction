# Predictive Modeling of Hospital Length of Stay and Discharge Type

## 1. Overview
This project aims to predict hospital length of stay (LOS) and discharge type using initial lab values, patient demographic data, and primary diagnosis information. These predictions can support clinical staff in decision-making and help improve hospital resource planning.

## 2. Structure
The repository is organized as follows:
- scripts/: Folder containing R scripts for data processing, modeling, and validation.
- notebooks/: RMarkdown files detailing each step of the analysis, from data exploration to modeling. --> planned
- reports/: Contains final reports, plots, and figures summarizing the findings. --> pending

## 3. Datasets
- **Lab Data**: Contains patient lab test results, sourced from RITM0154633_lab.csv. This dataset provides numerical and categorical lab values essential for analyzing patient conditions.
- **Clinical Data**: Contains patient demographics (e.g., age, sex), clinical details (e.g., discharge type, length of stay), and primary diagnoses. This data, sourced from RITM0154633_main.csv, will be used to explore patient outcomes and connect clinical characteristics with lab results.

## 4. Methodology
This analysis follows a structured approach to prepare, clean, and analyze data for building predictive models.

#### 4.1 Data Exploration
Initial examination of both datasets to understand their structure, contents, and any potential issues, such as missing values or duplicates. This step helps to identify data quality issues and provides insights into variable distributions.

#### 4.2 Data Cleaning
In this step, we address any identified issues, such as filling missing values, removing duplicates, and correcting data types. Cleaning ensures the data is accurate and ready for analysis.

#### 4.3 Data Merging
Lab and clinical datasets are merged using unique patient and case IDs to link test results with clinical information. This combined dataset allows for a more comprehensive analysis by connecting lab values with patient outcomes.

#### 4.4 Modeling Approaches (Planned)
Two predictive models are planned:

- Predicting Length of Stay: Linear Regression will be used to estimate the length of stay based on lab values, patient demographics, and diagnoses.
- Predicting Discharge Type: Logistic Regression will classify discharge type (e.g., home vs. non-home) using lab values and demographic information. These models aim to offer insights into how different factors influence patient outcomes.
Once the models are built and evaluated, more information will be provided.

#### 4.5 Model Validation (Planned)
Model performance will be evaluated using cross-validation and metrics like accuracy, sensitivity, specificity, and AUC. Statistical tests will also help confirm model reliability. This section will be updated with metrics once validation is complete.

## 5. Results (Pending)
This section will be populated with findings as the project progresses. It will include:
- Model performance
- Visualization
- Key insights
  
## 6. Future works

## 7. Reference papers
- “A Comparative Study of Pattern Recognition Algorithms for Predicting the Inpatient Mortality Risk Using Routine Laboratory Measurements” (https://doi.org/10.1007/s10462-018-9625-3)
- “Machine Learning Prediction of Hypoglycemia and Hyperglycemia From Electronic Health Records” (doi:10.2196/36176)

## 8. How to run this project
This section will explain the steps to run the project.
