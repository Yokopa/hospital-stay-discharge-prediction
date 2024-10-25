#===============================================================================
# Script: 03_merge_data.R
# Purpose: This script merges laboratory and clinical datasets, aggregates lab test results
#          to mean values for duplicate tests, transforms the data into a wide format,
#          and categorizes patients based on pseudo_patient_id.
# Data Source: Insel Data Science Center (IDSC)
# Author: Anna Scarpellini Pancrazi
# Date: 2024
# Input Files: 
#  - cleaned_lab_data.csv
#  - cleaned_clinical_data.csv
# Output File: merged_data.csv
#===============================================================================

# Packages
library(dplyr)
library(tidyr)
library(data.table)
setDTthreads()  # Use multiple threads for data.table operations

# Parameters
lab_data_path <- "./output_files/cleaned_lab_data.csv"
cleaned_data_path <- "./output_files/cleaned_clinical_data.csv"
merged_data_path <- "./output_files/merged_data.csv"

# Number of most common tests to keep
n_common_tests <- 100  # Set this to the number of common tests you want

#===============================================================================
# 1. Load the input data
#===============================================================================
lab_data <- read.csv(lab_data_path)
clinical_data <- read.csv(cleaned_data_path)

# View first few rows and summaries
head(lab_data)
head(clinical_data)

# Count unique case IDs and patient IDs in both datasets
unique_case_ids_lab <- uniqueN(lab_data$pseudo_case_id)
unique_patient_ids_lab <- uniqueN(lab_data$pseudo_patient_id)
cat("Unique case IDs in lab data:", unique_case_ids_lab, "\n")
cat("Unique patient IDs in lab data:", unique_patient_ids_lab, "\n")

unique_case_ids_clinical <- uniqueN(clinical_data$pseudo_case_id)
unique_patient_ids_clinical <- uniqueN(clinical_data$pseudo_patient_id)
cat("Unique case IDs in clinical data:", unique_case_ids_clinical, "\n")
cat("Unique patient IDs in clinical data:", unique_patient_ids_clinical, "\n")

# Count unique test_abbr in lab dataset
unique_test_abbr_lab <- uniqueN(lab_data$test_abbr)
cat("Unique test abbreviations in lab data:", unique_test_abbr_lab, "\n")

#===============================================================================
# 2. Check for duplicates before merging
#===============================================================================
lab_data_dt <- as.data.table(lab_data)
clinical_data_dt <- as.data.table(clinical_data)

# Check for duplicates in lab data
lab_duplicates <- lab_data_dt[duplicated(lab_data_dt) | duplicated(lab_data_dt, fromLast = TRUE)]
if (nrow(lab_duplicates) > 0) {
  cat("\nDuplicates found in lab data (first few rows):\n")
  print(head(lab_duplicates))
} else {
  cat("\nNo duplicates found in lab data.\n")
}

# Check for duplicates in clinical data
clinical_duplicates <- clinical_data_dt[duplicated(clinical_data_dt) | duplicated(clinical_data_dt, fromLast = TRUE)]
if (nrow(clinical_duplicates) > 0) {
  cat("\nDuplicates found in clinical data (first few rows):\n")
  print(head(clinical_duplicates))
} else {
  cat("\nNo duplicates found in clinical data.\n")
}

#===============================================================================
# 3. Filter lab dataset for tests with numeric results (first draft)
#===============================================================================
filtered_lab_data <- lab_data_dt[!is.na(num_result)]

# Count frequency of each test_abbr
test_counts <- filtered_lab_data %>%
  count(test_abbr, sort = TRUE)

# Select the top n_common_tests most frequent test_abbr
common_tests <- test_counts %>%
  top_n(n_common_tests) %>%
  pull(test_abbr)

# Filter the lab data to keep only the most common tests
filtered_lab_data <- filtered_lab_data %>%
  filter(test_abbr %in% common_tests)

cat("\nFiltered lab data after selecting the most common tests (first few rows):\n")
print(head(filtered_lab_data))

#===============================================================================
# 3. Filter lab dataset for tests with numeric results (second draft)
#===============================================================================
filtered_lab_data <- lab_data_dt[!is.na(num_result)]

# Count frequency of each test_abbr, considering only unique tests per case_id
unique_test_counts <- filtered_lab_data %>%
  distinct(pseudo_case_id, test_abbr) %>%
  count(test_abbr, name = "test_count") %>%
  arrange(desc(test_count))

# Select the top 100 most common test_abbr
common_tests <- unique_test_counts %>%
  top_n(100, test_count) %>%
  pull(test_abbr)

# Filter the lab data to keep only the most common tests
filtered_lab_data <- filtered_lab_data %>%
  filter(test_abbr %in% common_tests)

#===============================================================================
# 4. Remove the text_result column from the filtered lab data
#===============================================================================
filtered_lab_data[, text_result := NULL]  # Remove the text_result column

# Check the structure after removing text_result
cat("\nFiltered lab data after removing text_result (first few rows):\n")
print(head(filtered_lab_data))

#===============================================================================
# 5. Aggregate to mean values
#===============================================================================
# Aggregating lab_data to take mean for duplicate test_abbr within each pseudo_case_id
aggregated_lab_data_dt <- filtered_lab_data[, .(mean_result = mean(num_result, na.rm = TRUE)), 
                                            by = .(pseudo_patient_id, pseudo_case_id, test_abbr)]

#===============================================================================
# 6. Transform to wide format
#===============================================================================
# Transforming aggregated_lab_data to wide format
wide_lab_data_dt <- dcast(aggregated_lab_data_dt, 
                          pseudo_patient_id + pseudo_case_id ~ test_abbr, 
                          value.var = "mean_result")

#===============================================================================
# 7. Merge datasets
#===============================================================================
merged_data <- clinical_data_dt %>%
  select(pseudo_patient_id, pseudo_case_id, discharge_type, sex, age, length_of_stay_days, principal_diagnosis) %>%
  inner_join(wide_lab_data_dt, by = c("pseudo_patient_id", "pseudo_case_id"))

# Check the merged data
head(merged_data)
str(merged_data)
#summary(merged_data)

#===============================================================================
# 8. Categorically cluster patients
#===============================================================================
# Create patient_cluster as a factor and drop pseudo_patient_id
merged_data <- merged_data %>%
  mutate(patient_cluster = as.factor(pseudo_patient_id)) %>%  # Create patient_cluster
  select(-pseudo_patient_id)  # Remove the original pseudo_patient_id

# Check the structure after clustering
cat("\nStructure of merged data after clustering:\n")
str(merged_data)

#===============================================================================
# 9. Save merged data to CSV
#===============================================================================
write.csv(merged_data, merged_data_path, row.names = FALSE)
cat("Merged data saved to:", merged_data_path, "\n")

#===============================================================================
# Summary of Cleaning Steps
# This script has successfully:
# - Handled missing values in 'num_result' by filtering.
# - Checked for and identified duplicates in both lab and clinical datasets.
# - Selected the most common lab tests based on frequency.
# - Removed unnecessary text_result columns from the lab dataset.
# - Aggregated lab test results to mean values for duplicates.
# - Merged clinical and lab datasets into a comprehensive dataset.
# - Categorized patients based on their pseudo_patient_id.
# - Saved the final merged dataset to 'merged_data.csv' for further analysis.
#===============================================================================
