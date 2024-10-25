#===============================================================================
# Script: 01a_load_lab_data.R
# Purpose: Load and inspect the lab raw data to understand its structure,
#          translate column names from German to English, and extract unique 
#          test abbreviations and test names.
# Data Source: Insel Data Science Center (IDSC)
# Author: Anna Scarpellini Pancrazi
# Date: 2024
# Input File: RITM0154633_del_20240923/RITM0154633_lab.csv
# Output Files:
#   - lab_data_eng_cols.csv (lab data with translated English column names)
#   - unique_test_abbreviations.csv (unique test abbreviations sorted alphabetically)
#   - unique_test_names.csv (unique test names sorted alphabetically)
#   - - unique_test_name_abbr_pairs.csv (unique test_name and test_abbr pairs sorted alphabetically)
#===============================================================================

# Packages
library(data.table)
setDTthreads()

# Parameters
# Input file path for the lab data
lab_data_path <- "./RITM0154633_del_20240923/RITM0154633_lab.csv"

#===============================================================================
# 1. Load the lab data
#===============================================================================
lab_data <- read.csv(lab_data_path)

#===============================================================================
# 2. Basic data inspection
#===============================================================================
# View first few rows
head(lab_data)

# Column names
names(lab_data)

# Translate columns for non-German speakers:
colnames(lab_data) <- c("pseudo_patient_id", "pseudo_case_id", "test_name", 
                         "test_abbr", "method_num", "num_result", 
                         "text_result", "unit")

#===============================================================================
# 3. Data dimensions and overview
#===============================================================================
# Number of rows (observations) and columns (variables)
dim(lab_data)

# Column names
names(lab_data)

# View structure of the dataset (column types, number of observations, etc.)
str(lab_data)

# Get summary statistics for all columns
summary(lab_data)

#===============================================================================
# 4. Additional data checks
#===============================================================================
# Check for missing values
missing_values <- sapply(lab_data, function(x) sum(is.na(x)))
print(missing_values)

# Get unique values for all columns and count them
unique_values_list <- lapply(lab_data, unique)
unique_counts <- sapply(unique_values_list, length)
unique_df <- data.frame(
  Unique_Count = unique_counts,
  Unique_Values = I(unique_values_list)  # Using I() to prevent auto-unlisting
)
print(unique_df)

# Get unique test abbreviations, sort them, and display the result
unique_test_abbr <- sort(unique(lab_data$test_abbr))
unique_abbr_df <- data.frame(Unique_Test_Abbreviations = unique_test_abbr)
head(unique_abbr_df)

# Get unique test names, sort them, and display the result
unique_test_names <- sort(unique(lab_data$test_name))
unique_names_df <- data.frame(Unique_Test_Names = unique_test_names)
head(unique_names_df)

# Create unique test_name and test_abbr pairs
# Get unique pairs of test_name and test_abbr
unique_test_pairs <- unique(lab_data[, c("test_name", "test_abbr")])
# Sort the pairs by test_name for better readability
unique_test_pairs_sorted <- unique_test_pairs[order(unique_test_pairs$test_name), ]
# View the first few rows of the sorted unique test pairs
head(unique_test_pairs_sorted)

# Check for duplicate rows in the dataset
lab_data_dt <- as.data.table(lab_data) # Convert to data.table
# Find the duplicate rows
duplicates <- lab_data_dt[duplicated(lab_data_dt) | duplicated(lab_data_dt, fromLast = TRUE)]
# Count the number of duplicates
duplicates_count <- nrow(duplicates)
print(paste("Number of duplicate rows:", duplicates_count))
# Print the duplicate rows
print(duplicates)

#===============================================================================
# 5. Save any initial inspection results
#===============================================================================
# Save the lab data with English column names
write.csv(lab_data, "output_files/lab_data_eng_cols.csv", row.names = FALSE)

# Save column names in a CSV file
# write.csv(names(lab_data), "lab_data_column_names.csv", row.names = FALSE)

# Save unique_test_abbreviations in a CSV file
write.csv(unique_abbr_df, "output_files/unique_test_abbreviations.csv", row.names = FALSE)

# Save unique_test_names in a CSV file
write.csv(unique_names_df, "output_files/unique_test_names.csv", row.names = FALSE)

# Save unique test_name and test_abbr pairs to a CSV file
write.csv(unique_test_pairs_sorted, "output_files/unique_test_name_abbr_pairs.csv", row.names = FALSE)

#===============================================================================
# Summary of Script Actions
# This script has successfully:
# - Loaded the lab raw data and inspected its structure.
# - Translated column names from German to English.
# - Provided an overview of data dimensions, types, and summary statistics.
# - Checked for missing values and identified duplicates.
# - Saved the cleaned lab data with translated column names and unique test 
#   abbreviations and test names to specified output files.
# Output Files:
#   - lab_data_eng_cols.csv (lab data with translated English column names)
#   - unique_test_abbreviations.csv (unique test abbreviations sorted alphabetically)
#   - unique_test_names.csv (unique test names sorted alphabetically)
#   - unique_test_name_abbr_pairs.csv (unique test_name and test_abbr pairs sorted alphabetically)
#===============================================================================
