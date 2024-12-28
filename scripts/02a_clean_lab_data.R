#===============================================================================
# Script: 02a_clean_lab_data.R
# Purpose: Clean the lab data by handling missing values, 
#          correcting for duplicates, and removing rows with 
#          both num_result and text_result missing.
# Data Source: Insel Data Science Center (IDSC)
# Author: Anna Scarpellini Pancrazi
# Date: 2024
# Input File: lab_data_eng_cols.csv
# Output File: first_cleaned_lab_data.csv
#===============================================================================

# Packages
library(dplyr)   # For data manipulation
library(data.table)
setDTthreads(parallel::detectCores())

# Parameters
# Input file path for the lab data
lab_data_path <- "./output_files/lab_data_eng_cols.csv"
# Output file path for the cleaned lab data
cleaned_lab_data_path <- "./output_files/cleaned_lab_data.csv"

#===============================================================================
# 1. Load the lab data
#===============================================================================
lab_data <- read.csv(lab_data_path)

# View first few rows
head(lab_data)

summary(lab_data)

#===============================================================================
# 2. Handle Missing Data
#===============================================================================
# Count missing values in both datasets
missing_lab_data <- colSums(is.na(lab_data))

# Print missing value summaries
print(missing_lab_data)

  # 387136 missing entries for 'test_abbr'  (1)
  # 39 missing entries for 'text_result'    (2)

# (1) Filter rows with missing data - test_abbr
missing_test_abbr_rows <- lab_data %>% filter(is.na(test_abbr))

# (1) Print rows with missing test_abbr values
head(missing_test_abbr_rows)

# (1) Check if all missing test_abbr rows have 'Natrium' as value in the test_name column
all_missing_test_abbr_natrium <- all(missing_test_abbr_rows$test_name == "Natrium")
print(all_missing_test_abbr_natrium)

# (1) Fill missing test_abbr with 'NAT' where test_name is 'Natrium'
lab_data <- lab_data %>%
  mutate(test_abbr = if_else(is.na(test_abbr) & test_name == "Natrium", "NAT", test_abbr))

# (1) Check missing data again after changes
print(colSums(is.na(lab_data)))

# (2) Filter rows with missing data
missing_text_result_rows <- lab_data %>% filter(is.na(text_result))

# (2) Print rows with missing text_result values
# Just 39 so print them all
print(missing_text_result_rows) 

# All entries seem related to administrative or signature-related fields, 
# rather than quantitative results from lab tests. -> removal

# Remove rows with missing text_result
lab_data <- lab_data %>%
  filter(!is.na(text_result))

print(colSums(is.na(lab_data)))

# CLeanup
rm(missing_test_abbr_rows, missing_text_result_rows)
gc()

#===============================================================================
# 3. Remove duplicate rows
#===============================================================================
lab_data_dt <- as.data.table(lab_data) # Convert to data.table

# Find the duplicate rows
duplicates <- lab_data_dt[duplicated(lab_data_dt) | duplicated(lab_data_dt, fromLast = TRUE)]

# Count the number of duplicates before cleaning
duplicates_count <- nrow(duplicates)

print(paste("Number of duplicate rows before cleaning:", duplicates_count))
# Print the duplicate rows
print(duplicates)

# Remove duplicates and keep only unique rows
lab_data_dt_clean <- unique(lab_data_dt)

# Check how many rows were removed
duplicates_removed <- nrow(lab_data_dt) - nrow(lab_data_dt_clean)
print(paste("Number of duplicates removed:", duplicates_removed))

# Print the number of duplicated rows after cleaning
print(lab_data_dt_clean[duplicated(lab_data_dt_clean) | duplicated(lab_data_dt_clean, fromLast = TRUE)])

# Cleanup
rm(duplicates, duplicates_count, duplicates_removed)
gc()

#===============================================================================
# 4. Check plausibility of num_result values
#===============================================================================
# Check for rows with missing, special characters, or unexpected values

# Identify and count non-numeric values (e.g., non-numeric strings, special placeholders)
non_numeric_entries <- lab_data_dt_clean %>%
  filter(!grepl("^-?\\d*(\\.\\d+)?$", num_result) & num_result != "")
num_non_numeric_entries <- nrow(non_numeric_entries)
print(paste("Number of non-numeric entries found in num_result:", num_non_numeric_entries))

# Show a sample of non-numeric entries if there are any
if (num_non_numeric_entries > 0) {
  print("Sample of non-numeric entries:")
  print(head(non_numeric_entries, 10))  # Show the first 10 non-numeric entries
}

# Filter and check entries where num_result is not NULL or NA
non_null_entries <- non_numeric_entries %>%
  filter(!is.na(num_result) & num_result != "NULL")  # "NULL" is stored as a string 

# Show a sample of non-null entries
num_non_null_entries <- nrow(non_null_entries)
print(paste("Number of non-NULL entries found in num_result:", num_non_null_entries))

# If there are non-null entries, print a sample of them
if (num_non_null_entries > 0) {
  print("Sample of non-NULL entries in num_result:")
  print(head(non_null_entries, 10))  # Show the first 10 non-NULL entries
}

  # Explanation of non-numeric non-null entries:
  # These entries represent placeholder values or special messages triggered 
  # by specific conditions in the dataset. These conditions could include:
  # - Missing or incomplete data (e.g., results not available or pending).
  # - Errors in data collection or processing.
  # - Non-performed tests (e.g., tests that were ordered but not executed).
  # - Triggered messages based on specific logic (e.g., abnormal results, threshold exceedances).

# Replace NULL or empty values in num_result with NA
lab_data_dt_clean <- lab_data_dt_clean %>%
  mutate(num_result = ifelse(num_result == "NULL" | num_result == "", NA, num_result))

# Check for NaN, Inf, or -Inf values
nan_inf_values <- lab_data_dt_clean %>%
  filter(num_result %in% c("NaN", "Inf", "-Inf"))
print(paste("NaN/Inf entries:", nrow(nan_inf_values)))

# Check for empty strings
empty_values <- lab_data_dt_clean %>%
  filter(num_result == "")
print(paste("Empty string entries:", nrow(empty_values)))

# Convert num_result to numeric, setting non-numeric entries to NA
lab_data_dt_clean <- lab_data_dt_clean %>%
  mutate(num_result = as.numeric(num_result))

# Check for entries that were converted to NA after conversion
na_num_result_entries_after <- lab_data_dt_clean %>%
  filter(is.na(num_result))

# Print the non-numeric entries that were set to NA after conversion
if (nrow(na_num_result_entries_after) > 0) {
  print("Non-numeric entries found in num_result and converted to NA")
  print(paste("(", nrow(na_num_result_entries_after), " entries ):"))
  print(head(na_num_result_entries_after, 10))  # Show the first 10 entries that became NA
}

# Clean
rm(non_numeric_entries, non_null_entries, nan_inf_values, empty_values, na_num_result_entries_after)
gc()
   
#-------------------------------------------------------------------------------

# Get unique test names with negative num_result values
unique_negative_test_names <- lab_data_dt_clean %>%
  filter(num_result < 0) %>%
  pull(test_name) %>%
  unique()

# Print the unique test names with negative values
if (length(unique_negative_test_names) > 0) {
  print("Unique test names with negative num_result values:")
  print(unique_negative_test_names)
} else {
  print("No unique test names with negative num_result values found.")
}

# Calculate the percentage of negative values for each test, along with method_num
negative_percentage <- lab_data_dt_clean %>%
  group_by(test_name, method_num) %>%
  summarise(
    total_count = n(),
    negative_count = sum(num_result < 0, na.rm = TRUE),
    negative_percentage = (negative_count / total_count) * 100,
    .groups = 'drop'
  )%>%
  filter(negative_count > 0)

# Display the results
print(negative_percentage, n=65)

# Filter to keep only Basen-Excess variations or tests with negative values
lab_data_dt_clean <- lab_data_dt_clean %>%
  filter(num_result >= 0 | grepl("Basen-Excess", test_name))

# Clean
rm(unique_negative_test_names, negative_percentage)
gc()

#-------------------------------------------------------------------------------

# Remove values incompatible with life PART 1

# Define limits for incompatible values for specific tests
limits <- list(
  "Natrium" = c(100, 191),       # Sodium (mmol/L) - paper
  "Kalium" = c(1.2, 9.8),        # Potassium (mmol/L) - Ana
  "Chloride" = c(65, 138),       # Chloride (mmol/L) - paper
  "pH" = c(6.8, 7.8)            # pH (both venous and arterial) - Ana
)

# Function to apply the limit checks using test_abbr
filter_incompatible_values <- function(data, test_abbr, lower_bound, upper_bound) {
  # Identify the entries to remove
  removed_entries <- data %>%
    filter(test_abbr == !!test_abbr & (num_result < lower_bound | num_result > upper_bound))
  
  # Print the removed entries
  if (nrow(removed_entries) > 0) {
    print(paste("Removed entries for test:", test_abbr))
    print(removed_entries)
  }
  
  # Filter out the incompatible values
  data %>%
    filter(!(test_abbr == !!test_abbr & (num_result < lower_bound | num_result > upper_bound)))
}

# Apply filtering for each test based on the defined limits
lab_data_dt_clean <- lab_data_dt_clean %>%
  filter_incompatible_values(., "NAT", limits$Natrium[1], limits$Natrium[2]) %>%  # Sodium
  filter_incompatible_values(., "KA", limits$Kalium[1], limits$Kalium[2]) %>%     # Potassium
  filter_incompatible_values(., "CL", limits$Chloride[1], limits$Chloride[2]) %>% # Chloride
  filter_incompatible_values(., "pH", limits$pH[1], limits$pH[2]) %>%             # Arterial pH
  filter_incompatible_values(., "pHa", limits$pH[1], limits$pH[2]) %>%            # Venous pH
  filter_incompatible_values(., "pHv", limits$pH[1], limits$pH[2])                # Unspecified pH

#-------------------------------------------------------------------------------

# Remove Incompatible Values for Life - Part 2

# After the manual exploration of the merged dataset with clinical data
# (in a later step, performed in script 04_merged_DE.R), I identified certain
# lab test entries that were inconsistent or incompatible with life. As a result,
# I am going back to the cleaning process to remove these specific entries.

###########
#   Hbn   #
###########
# Values of zero for hemoglobin are not compatible with life.
# Furthermore, the corresponding diagnoses do not provide a plausible explanation
# for such low values. These are likely errors and should be removed from the analysis.
# Remove rows where Hbn (hemoglobin) value is 0

lab_data_dt_clean <- lab_data_dt_clean %>%
  filter(!(test_name == "Hbn" & num_result == 0))

############
#   Eryn   #
############
# Remove rows where Eryn (erythrocytes) value is 0
lab_data_dt_clean <- lab_data_dt_clean %>%
  filter(!(test_name == "Eryn" & num_result == 0))

############
#    Hkn   #
############
# Remove rows where Hkn value is greater than 1
lab_data_dt_clean <- lab_data_dt_clean %>%
  filter(!(test_name == "Hkn" & num_result > 1))

###########
#   INR   #
###########
# Remove rows where INR value is 9999 or 5001.36 (likely placeholders or errors)
lab_data_dt_clean <- lab_data_dt_clean %>%
  filter(!(test_name == "INR" & num_result %in% c(9999, 5001.36)))

############
#   QUHD   #
############
# Remove rows where QUHD (likely a test for glucose or other parameters) value is 9999.00
lab_data_dt_clean <- lab_data_dt_clean %>%
  filter(!(test_name == "QUHD" & num_result == 9999.00))

###########
#   CA    #
###########
# Remove rows where CA (calcium) value is 0
lab_data_dt_clean <- lab_data_dt_clean %>%
  filter(!(test_name == "CA" & num_result == 0))

############
#   Tbga   #
############
# Remove rows where Tbga_max_value (likely a temperature) is 34,103°C (which is highly unrealistic)
lab_data_dt_clean <- lab_data_dt_clean %>%
  filter(!(test_name == "Tbga_max_value" & num_result == 34103))

############
#   pCO2   #
############
# Remove rows where pCO2 value is 200 (unrealistically high)
lab_data_dt_clean <- lab_data_dt_clean %>%
  filter(!(test_name == "pCO2" & num_result == 200))

###########
#   pO2   #
###########
# Remove rows where pO2 value is 0 (impossible or unrealistic value for oxygen saturation)
lab_data_dt_clean <- lab_data_dt_clean %>%
  filter(!(test_name == "pO2" & num_result == 0))

# Clean up memory after filtering
gc()

#===============================================================================
# 5. Clean text_result
#===============================================================================
# Check for numeric entries in text_result before replacing them with NA
numeric_text_result_entries <- lab_data_dt_clean %>%
  filter(grepl("^[0-9]+$", text_result))

# Print numeric entries found in text_result
if (nrow(numeric_text_result_entries) > 0) {
  print("Numeric entries found in text_result:")
  print(numeric_text_result_entries)
} else {
  print("No numeric entries found in text_result.")
}

# Convert numeric entries in text_result to NA
lab_data_dt_clean <- lab_data_dt_clean %>%
  mutate(text_result = ifelse(grepl("^[0-9]+$", text_result), NA, text_result))

# Check for entries that were set to NA in text_result
na_text_result_entries <- lab_data_dt_clean %>%
  filter(is.na(text_result))

# Print entries that were set to NA in text_result
if (nrow(na_text_result_entries) > 0) {
  print("Numeric entries found in text_result and removed:")
  print(na_text_result_entries)
}

# Summary of the cleaned dataset after conversions
summary(lab_data_dt_clean)

#-------------------------------------------------------------------------------

# See which rows had "NULL" before replacing them
null_entries <- lab_data_dt_clean %>%
  filter(text_result == "NULL")

if (nrow(null_entries) > 0) {
  print("Entries found in text_result with 'NULL':")
  print(null_entries)
}

# Handle entries with "NULL" in text_result
lab_data_dt_clean <- lab_data_dt_clean %>%
  mutate(text_result = ifelse(text_result == "NULL", NA, text_result))

# After replacing "NULL" with NA, check for entries that were set to NA in text_result
na_text_result_entries <- lab_data_dt_clean %>%
  filter(is.na(text_result))

# Print entries that were set to NA in text_result
if (nrow(na_text_result_entries) > 0) {
  print("Entries found in text_result and converted to NA:")
  print(na_text_result_entries)
}

# Summary of the cleaned dataset after conversions
summary(lab_data_dt_clean)
str(lab_data_dt_clean)

#===============================================================================
# 6. Remove rows where both num_result and text_result are missing
#===============================================================================
head(lab_data_dt_clean %>%
       filter((is.na(num_result) & is.na(text_result)))
)

lab_data_dt_clean <- lab_data_dt_clean %>%
  filter(!(is.na(num_result) & is.na(text_result)))

# Check the number of rows after removal
print(paste("Number of rows after removing rows with both num_result and text_result missing:", nrow(lab_data_dt_clean)))

#===============================================================================
# 7. Final check for duplicates and summary of the cleaned dataset
#===============================================================================
# Check for duplicates before saving the final cleaned dataset
final_duplicates <- lab_data_dt_clean[duplicated(lab_data_dt_clean) | duplicated(lab_data_dt_clean, fromLast = TRUE)]

if (nrow(final_duplicates) > 0) {
  print("Duplicates found before saving the final data:")
  print(final_duplicates)
  
  # Check if all duplicates have test_name == "Benutzer" and num_result == 0
  all_benutzer_and_zero <- all(final_duplicates$test_name == "Benutzer" & final_duplicates$num_result == 0)
  
  if (all_benutzer_and_zero) {
    # If all duplicates match the condition, remove them
    lab_data_dt_clean <- lab_data_dt_clean %>%
      filter(!(duplicated(lab_data_dt_clean) | duplicated(lab_data_dt_clean, fromLast = TRUE)) | 
               !(test_name == "Benutzer" & num_result == 0))
    
    print("All duplicates had 'Benutzer' as test_name and num_result = 0, and they were removed.")
  } else {
    print("Not all duplicates had 'Benutzer' as test_name and num_result = 0.")
  }
}

# Remove rows where test_name == "Benutzer" and num_result == 0
lab_data_dt_clean <- lab_data_dt_clean[!(test_name == "Benutzer" & num_result == 0)]

# Check for remaining duplicates after the removal
final_duplicates <- lab_data_dt_clean[duplicated(lab_data_dt_clean) | duplicated(lab_data_dt_clean, fromLast = TRUE)]

if (nrow(final_duplicates) > 0) {
  print("Duplicates found after removing 'Benutzer' rows with num_result = 0:")
  print(head(final_duplicates))  # Display the first few duplicates for inspection
  
  # Remove remaining duplicates
  lab_data_dt_clean <- unique(lab_data_dt_clean)
  
  print("Remaining duplicates were removed.")
} else {
  print("No duplicates found after removing 'Benutzer' rows with num_result = 0.")
}

# Last check for duplicates
print(lab_data_dt_clean[duplicated(lab_data_dt_clean) | duplicated(lab_data_dt_clean, fromLast = TRUE)])

#-------------------------------------------------------------------------------

# Final row count after cleaning
print(paste("Number of rows after cleaning:", nrow(lab_data_dt_clean)))

summary(lab_data_dt_clean)
str(lab_data_dt_clean)

#-------------------------------------------------------------------------------

# Final check for missing values
missing_lab_data_clean <- colSums(is.na(lab_data_dt_clean))
# Print missing value summaries
print(missing_lab_data_clean)

#===============================================================================
# 8. Save the cleaned lab data
#===============================================================================
write.csv(lab_data_dt_clean, cleaned_lab_data_path, row.names = FALSE)

#===============================================================================
# Summary of Cleaning Steps
# This script has successfully:
# - Handled missing values in test_abbr and text_result
# - Removed duplicate rows from the dataset
# - Converted 'num_result' to numeric and removed non-numeric entries, replacing them with NA.
# - Converted numeric entries in 'text_result' to NA.
# - Removed rows where both 'num_result' and 'text_result' are missing.
# - Rechecked for any remaining duplicates before saving the final cleaned data
# - Saved the cleaned lab data to 'cleaned_lab_data.csv' for further analysis
#===============================================================================