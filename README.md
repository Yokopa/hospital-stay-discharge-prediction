# Predictive Modeling of Hospital Length of Stay and Discharge Type

## 1. Overview
This project aims to predict hospital length of stay (LOS) and discharge type using initial lab values, patient demographic data, and primary diagnosis information. The predictive models will help provide actionable insights to clinical staff and improve hospital resource management.

## 2. Structure
The repository is organized as follows:
- scripts/: Folder containing R scripts for data processing, modeling, and validation.
- notebooks/: RMarkdown files detailing each step of the analysis, from data exploration to modeling. --> planned
- reports/: Contains final reports, plots, and figures summarizing the findings. --> pending

## 3. Datasets
- **Lab Data**: Sourced from a CSV file (RITM0154633_lab.csv), which ontains patient lab results.
- **Clinical Data**: Sourced from a CSV file (RITM0154633_main.csv), which includes patient demographics (age, sex), clinical details (discharge type, length of stay), and primary diagnosis information.

## 4. Methodology
The following steps were taken in the analysis:

#### 4.1 Data Exploration
- The lab data dataset contains 20,487,004 observations and 8 variables, with a file size of 1.0 GB. It has 0 missing values for most columns, but the test_abbr column has 387,136 missing entries, and the text_result column has 39 missing values. Additionally, there are 25 duplicate rows identified within the dataset. The dataset includes 182,238 unique patients and 311,611 unique case identifiers, along with 3,470 unique test names and 4,609 unique test abbreviations.

- The clinical data dataset consists of 311,629 observations and 7 variables, with a file size of 12.3 MB. Most columns have 0 missing values, except for the discharge_type column, which has 2 missing entries. There are 0 duplicate rows identified within the dataset. The dataset includes 182,242 unique patients and 311,629 unique case identifiers, along with 6,810 unique principal diagnoses. The dataset's demographic distribution shows 145,233 female and 166,396 male patients. Age statistics reveal a minimum age of -1 (indicating a possible data entry error), a maximum age of 108, and a mean age of approximately 55 years.
  
#### 4.2 Data Cleaning
- The lab data dataset originally consisted of 20,487,004 observations and 8 variables. Most columns had no missing values, except for the test_abbr column, which had 387,136 missing entries, and the text_result column, which had 39 missing entries. The missing test_abbr values were filled with "NAT" for rows where the test name was "Natrium." The missing text_result values were related to administrative fields and were removed. The dataset initially contained 50 duplicate rows, with 25 duplicates removed during the cleaning process. Non-numeric values in the num_result column were converted to NA, and valid numeric values were cast to the appropriate numeric type. Negative values were reviewed and removed, except for measurements related to "base excess," where negative values are permissible. Additionally, values generally considered incompatible with human life were filtered out specifically for a limited number of analytes, including natrium, potassium, chloride, and blood pH, focusing on those that fell outside clinically acceptable limits to maintain data integrity. All NULL values in the text_result column were replaced with NA to maintain consistency in handling missing data. Additionally, any numeric values that were incorrectly placed in the text_result column were replaced with NA, as they should have been recorded in the num_result column. Rows where both num_result and text_result were missing were also removed to ensure data quality. After the initial cleaning steps, a final check for duplicates was performed. All entries with the test name Benutzer and a num_result of 0 were removed because they were considered administrative data with no analytical value. For the remaining duplicate entries, one of each pair was retained. Following these actions, the final dataset contains 19,031,056 observations and 8 variables.

- The clinical dataset consisted of multiple variables, focusing on discharge type, patient age, and length of stay. The dataset had two missing entries in the discharge_type column. After analyzing the patients' histories and diagnoses (I25.19 and I48.1), the missing discharge types were removed due to their minimal impact on the dataset. Regarding age anomalies, one patient had an age of -1. The dataset contained 20,738 entries where age was equal to 0, corresponding to 18,268 unique patients. Given the significant number of entries for newborns, which were likely influenced by data entry errors, the decision was made to focus exclusively on adult patients by retaining only those aged 18 or older in the dataset. A scatter plot was generated to identify outliers in the length of stay variable. No substantial outliers were found.

#### 4.3 Data Merging
- In this merging step, the lab and clinical datasets were combined using unique patient and case IDs to link test results with clinical information. A key challenge was the large size of the lab dataset, which caused memory overload when transforming the data into a wide format. Although no duplicate records were found, the main issue was handling thousands of lab tests efficiently (**3,896** unique test abbreviations). To address this, the script filtered the lab data to include only the most common tests to reduce memory usage. However, this approach may not fully capture the complexity of patient cases and could overlook important but less frequent tests.
The merging was performed using an **inner join** based on the `pseudo_patient_id` and `pseudo_case_id` columns, ensuring that only records present in both datasets were retained. 
After merging, patients were categorized into clusters for analysis by creating a new factor variable, `patient_cluster`, based on `pseudo_patient_id`. The original `pseudo_patient_id` column was then removed to avoid redundancy, allowing for a clearer focus on case-level analysis. This step facilitates further analysis while maintaining the independence of each case ID.

#### 4.4 Modeling Approaches (Planned)
- Predicting Length of Stay: Linear Regression to model LOS based on lab values, patient demographics, and diagnoses.
- Predicting Discharge Type: Logistic Regression to classify discharge type (e.g., home vs. non-home) based on initial lab values and patient demographic information.

Once the models are built and evaluated, more information will be provided.

#### 4.5 Model Validation (Planned)
Model validation will be performed using various techniques, including:
- Cross-validation techniques
- Performance metrics (accuracy, sensitivity, specificity, AUC, etc.)
- Statistical tests

This section will be updated with results and performance metrics after validation is completed.

## 5. Results (Pending)
This section will be filled in as the project progresses and results become available. The following will be included:
- Model performance
- Visualization
- Key insights
  
## 6. Future works

## 7. Reference papers
- “A Comparative Study of Pattern Recognition Algorithms for Predicting the Inpatient Mortality Risk Using Routine Laboratory Measurements” (https://doi.org/10.1007/s10462-018-9625-3)
- “Machine Learning Prediction of Hypoglycemia and Hyperglycemia From Electronic Health Records” (doi:10.2196/36176)

## 8. How to run this project
This section will explain the steps to run the project.
