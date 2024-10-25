#===============================================================================
# Script: 02b_clean_clinical_data.R
# Purpose: Clean the clinical data by handling missing values 
#          and correcting age entries.
# Data Source: Insel Data Science Center (IDSC)
# Author: Anna Scarpellini Pancrazi
# Date: 2024
# Input File: clinical_data_eng_cols.csv
# Output File: cleaned_clinical_data.csv
#===============================================================================

# Packages
library(dplyr)  
library(ggplot2) 

# Parameters
# Input file path for the lab data
clinical_data_path <- "./output_files/clinical_data_eng_cols.csv"
# Output file path for the cleaned lab data
cleaned_clinical_data_path <- "./output_files/cleaned_clinical_data.csv"

#===============================================================================
# 1. Load the lab data
#===============================================================================
clinical_data <- read.csv(clinical_data_path)

# View first few rows
head(clinical_data)

#===============================================================================
# 2. Handle missing data
#===============================================================================
# Count missing values in both datasets
missing_clinical_data <- colSums(is.na(clinical_data))

# Print missing value summaries
print(missing_clinical_data)

  # 2 missing entries for 'discharge_type'

# Filter rows with missing data
missing_discharge_rows <- clinical_data %>% filter(is.na(discharge_type))

# Print rows with missing data
print(missing_discharge_rows)

    # Just two rows from two different patients

# Filter the clinical data to find other entries for those two patients
missing_discharge_patient_1 <- clinical_data %>% filter(pseudo_patient_id == 187867)
missing_discharge_patient_1

missing_discharge_patient_2 <- clinical_data %>% filter(pseudo_patient_id == 235389)
missing_discharge_patient_2

# Summarize discharge variation separately for patients with I25.19 and I48.1 diagnoses
discharge_summary_by_diagnosis <- clinical_data %>%
  filter(principal_diagnosis %in% c("I21.4", "I25.19", "I48.1")) %>%
  group_by(principal_diagnosis, pseudo_patient_id) %>%
  summarise(
    unique_discharges = n_distinct(discharge_type, na.rm = TRUE),
    discharge_types = paste(unique(discharge_type, na.rm = TRUE), collapse = ", "),
    .groups = 'drop'
  )

# Count how many patients had only 1 discharge and how many had more than 1, by diagnosis
discharge_change_summary_by_diagnosis <- discharge_summary_by_diagnosis %>%
  mutate(discharge_change = ifelse(unique_discharges == 1, "No Change", "Change")) %>%
  group_by(principal_diagnosis, discharge_change) %>%
  summarise(
    count = n(),
    discharge_changes = paste(unique(discharge_types[unique_discharges > 1]), collapse = " / "),
    .groups = 'drop'
  )

# View the result
print(discharge_change_summary_by_diagnosis)

# Find the most common discharge types for diagnosis I25.19
common_discharges_I25_19 <- clinical_data %>%
  filter(principal_diagnosis == "I25.19") %>%
  group_by(discharge_type) %>%
  summarise(count = n(), .groups = 'drop') %>%
  mutate(percentage = (count / sum(count)) * 100) %>%  # Calculate percentage
  arrange(desc(count))  # Sort by count in descending order

# View the result
print(common_discharges_I25_19)

# View the most common discharge types
print(common_discharges_I25_19)

# ------------------------------------------------------------------------------

# APPROACH 1: IMPUTATION

  # For patient1 (id: 18786), considering that the most common discharge type for diagnosis I125.19 is 'Entlassung',
  # which is also the discharge type of the previous entry, it's safe to impute the missing entry 
  # with 'Entlassung',

  # For patient2 (id: 235389) , considering that only for 1% of the patients with diagnosis I48.1 
  # their discharge type changed over time,
  # so it's safe to impute the discharge type based on the previous entry ('Entlassung').

# Impute missing discharge types for patients based on the previous entry
#clinical_data_imputation <- clinical_data %>%
#  group_by(pseudo_patient_id) %>%
#  mutate(discharge_type = ifelse(is.na(discharge_type), 
#                                 lag(discharge_type, order_by = pseudo_case_id, default = "Entlassung"), 
#                                 discharge_type)) %>%
#  ungroup()

# View the updated data with imputed discharge types
#print(clinical_data_imputation %>% filter(pseudo_patient_id %in% c(187867, 235389)))

# Check missing data again after changes
#print(colSums(is.na(clinical_data_imputation)))
# ------------------------------------------------------------------------------

# APPROACH 2: REMOVAL -> chosen approach 
  # Removing 2 missing entries has negligible impact on the large dataset (311,629 obs) 
  # and simplifies analysis while reducing potential biases from imputation.

# Removing rows with missing discharge_type
clinical_data_cleaned <- clinical_data %>%
  filter(!is.na(discharge_type))

# Check if the rows have been removed
print(colSums(is.na(clinical_data_cleaned)))

#===============================================================================
# 3. Check and clean entries with invalid age
#===============================================================================
# Visualize age distribution before cleaning
ggplot(clinical_data_cleaned, aes(x = age)) +
  geom_histogram(binwidth = 1, fill = "skyblue", color = "black", alpha = 0.7) +
  labs(title = "Age distribution before cleaning", x = "Age", y = "Number of entries") +
  theme_minimal()

# Summarize data by counting unique patients for each age
age_distribution <- clinical_data_cleaned %>%
  group_by(age) %>%
  summarise(unique_patients = n_distinct(pseudo_patient_id))

# Plot the age distribution based on unique patients
ggplot(age_distribution, aes(x = age, y = unique_patients)) +
  geom_bar(stat = "identity", fill = "skyblue", color = "black", alpha = 0.7) +
  labs(title = "Age distribution by unique patients before cleaning", x = "Age", y = "Number of unique patients") +
  theme_minimal()

#-------------------------------------------------------------------------------

# Identify potential age errors
age_errors <- clinical_data_cleaned %>%
  filter(age < 0 | age > 120)
print(age_errors)

# Check other entries for the patient with negative age
patient_negative_age <- clinical_data_cleaned %>%
  filter(pseudo_patient_id == 39934)
print(patient_negative_age)

#-------------------------------------------------------------------------------
# Count entries where age is 1
count_age_one <- nrow(clinical_data_cleaned %>% filter(age == 1))
print(count_age_one)  # Actual output: 2224

# Count unique patients where age is 1
count_age_one_patients <- clinical_data_cleaned %>%
  filter(age == 1) %>%
  distinct(pseudo_patient_id) %>%
  nrow()
print(count_age_one_patients)

#-------------------------------------------------------------------------------

# Investigate rows where age is 0
age_zero_rows <- clinical_data_cleaned %>%
  filter(age == 0)
print(age_zero_rows)
print(nrow(age_zero_rows))

# Investigate unique patients where age is 0
age_zero_patients <- clinical_data_cleaned %>%
  filter(age == 0) %>%
  distinct(pseudo_patient_id)
#print(age_zero_patients)
print(nrow(age_zero_patients))  # Count of unique patients with age 0

#-------------------------------------------------------------------------------

# Now, keep only adult patients (age >= 18)
clinical_data_cleaned <- clinical_data_cleaned %>%
  filter(age >= 18)  # Focus on adult patients

# Visualize age distribution after cleaning
ggplot(clinical_data_cleaned, aes(x = age)) +
  geom_histogram(binwidth = 1, fill = "skyblue", color = "black", alpha = 0.7) +
  labs(title = "Age distribution of adult patients", x = "Age", y = "Number of entries") +
  theme_minimal()

# Check
print(clinical_data_cleaned %>%
        filter(age < 18 | age > 120))
#-------------------------------------------------------------------------------

# Summarize data by counting unique patients for each age
age_distribution <- clinical_data_cleaned %>%
  group_by(age) %>%
  summarise(unique_patients = n_distinct(pseudo_patient_id))

# Plot the age distribution based on unique patients
ggplot(age_distribution, aes(x = age, y = unique_patients)) +
  geom_bar(stat = "identity", fill = "skyblue", color = "black", alpha = 0.7) +
  labs(title = "Age distribution by unique adult patients", x = "Age", y = "Number of unique adult patients") +
  theme_minimal()

#===============================================================================
# 4. Check for outliers in length of stay
#===============================================================================
# Visualize length of stay against patient ID
ggplot(clinical_data_cleaned, aes(x = pseudo_patient_id, y = length_of_stay_days)) +
  geom_point(color = "skyblue", alpha = 0.7) +  # Adjust alpha for better visibility
  labs(title = "Length of Stay by Patient ID", x = "Patient ID", y = "Length of Stay (Days)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))  # Rotate x-axis labels for better readability

# Print summary statistics for length of stay
length_of_stay_summary <- summary(clinical_data_cleaned$length_of_stay_days)  
print(length_of_stay_summary)

#===============================================================================
# 3. Summary of the cleaned dataset
#===============================================================================
summary(clinical_data)

#===============================================================================
# 4. Save the cleaned clinical data
#===============================================================================
write.csv(clinical_data, cleaned_clinical_data_path, row.names = FALSE)

#===============================================================================
# Summary of Cleaning Steps
# This script has successfully:
# - Handled missing values in 'discharge_type' by removing entries.
# - Checked for and corrected invalid age entries.
# - Visualized key data distributions to identify outliers.
# - Saved the cleaned clinical data to 'cleaned_clinical_data.csv' for further analysis.
#===============================================================================
