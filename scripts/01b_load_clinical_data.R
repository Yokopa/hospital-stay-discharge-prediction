#===============================================================================
# Script: 01b_load_clinical_data.R
# Purpose: Load and inspect the clinical raw data to understand its structure, 
#          check for missing values and duplicates, and translate column names 
#          from German to English.
# Data Source: Insel Data Science Center (IDSC)
# Author: Anna Scarpellini Pancrazi
# Date: 2024
# Input File: RITM0154633_del_20240923/RITM0154633_main.csv
# Output Files: 
#   - clinical_data_eng_cols.csv (clinical data with translated column names)
#   - unique_diagnoses.csv (list of unique diagnoses)
#===============================================================================

# Parameters
# Input file path for the lab data
clinical_data_path <- "./RITM0154633_del_20240923/RITM0154633_main.csv"

#===============================================================================
# 1. Load the lab data
#===============================================================================
clinical_data <- read.csv(clinical_data_path)

#===============================================================================
# 2. Basic Data Inspection
#===============================================================================
# View first few rows
head(clinical_data)

# Column names
names(clinical_data)

# Translate columns for non-German speakers:
colnames(clinical_data) <- c("pseudo_patient_id", "pseudo_case_id", 
                             "discharge_type", "sex", "age", 
                             "length_of_stay_days", "principal_diagnosis")

#===============================================================================
# 3. Data Dimensions and Overview
#===============================================================================
# Number of rows (observations) and columns (variables)
dim(clinical_data)

# Column names
names(clinical_data)

# View structure of the dataset (column types, number of observations, etc.)
str(clinical_data)

# Get summary statistics for all columns
summary(clinical_data)

#===============================================================================
# 4. Additional data checks
#===============================================================================
# Check for missing values
missing_values <- sapply(clinical_data, function(x) sum(is.na(x)))
print("Missing values in clinical data:")
print(missing_values)

# Check for duplicate rows
duplicates_count <- sum(duplicated(clinical_data))
print(paste("Number of duplicate rows in clinical data:", duplicates_count))

# Unique Patients and Cases
unique_patients <- length(unique(clinical_data$pseudo_patient_id))
unique_cases <- length(unique(clinical_data$pseudo_case_id))
print(paste("Number of unique patients in clinical data:", unique_patients))
print(paste("Number of unique cases in clinical data:", unique_cases))

# Count of Male and Female
gender_counts <- table(clinical_data$sex)
print("Gender counts in clinical data:")
print(gender_counts)

# Age statistics
age_summary <- summary(clinical_data$age)
print("Age summary statistics:")
print(age_summary)

# Unique Diagnoses, sorted alphabetically
unique_diagnoses <- sort(unique(clinical_data$principal_diagnosis))
print(paste("Number of unique diagnoses in clinical data:", length(unique_diagnoses)))
unique_diagnoses_df <- data.frame(Unique_Diagnoses = unique_diagnoses)
print(head(unique_diagnoses_df))

# Unique Discharge Types
unique_discharge_types <- unique(clinical_data$discharge_type)
print(paste("Number of unique discharge types in clinical data:", length(unique_discharge_types)))
print(unique_discharge_types)

#===============================================================================
# 4. Save any initial inspection results
#===============================================================================
# Save the lab data with English column names
write.csv(clinical_data, "output_files/clinical_data_eng_cols.csv", row.names = FALSE)

# Save column names in a csv file:
# write.csv(names(clinical_data), "clinical_data_column_names.csv", row.names = FALSE)

# Save the unique diagnoses DataFrame to a CSV file
write.csv(unique_diagnoses_df, "output_files/unique_diagnoses.csv", row.names = FALSE)

#===============================================================================
# Summary of Script Actions
# This script has successfully:
# - Loaded the clinical raw data and inspected its structure.
# - Translated column names from German to English.
# - Provided an overview of data dimensions, types, and summary statistics.
# - Checked for missing values and identified duplicates.
# - Provided counts of unique patients, cases, genders, and diagnoses.
# - Saved the cleaned clinical data with translated column names and unique 
#   diagnoses to specified output files.
# Output Files:
#   - clinical_data_eng_cols.csv (clinical data with translated column names)
#   - unique_diagnoses.csv (list of unique diagnoses sorted alphabetically)
#===============================================================================

