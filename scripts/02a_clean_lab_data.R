#===============================================================================
# Script: 02a_clean_lab_data.R
# Purpose: Clean the lab data by handling missing values, 
#          correcting for duplicates, and removing rows with 
#          both num_result and text_result missing.
# Data Source: Insel Data Science Center (IDSC)
# Author: Anna Scarpellini Pancrazi
# Date: 2024
# Input File: lab_data_eng_cols.csv
# Output File: cleaned_lab_data.csv
#===============================================================================

# Packages
library(dplyr)   # For data manipulation
library(data.table)
setDTthreads()

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

#===============================================================================
# 4. Check plausibility of num_result values
#===============================================================================
# Count non-numeric entries in num_result
na_num_result_entries <- lab_data_dt_clean %>%
  filter(is.na(as.numeric(num_result)) & num_result != "")

num_non_numeric_entries <- nrow(na_num_result_entries)
print(paste("Number of non-numeric entries found in num_result:", num_non_numeric_entries))

# Show a sample if there are any
if (num_non_numeric_entries > 0) {
  print("Sample of non-numeric entries:")
  print(head(na_num_result_entries, 10))  # Show the first 10 entries
}

# Convert num_result to numeric, setting non-numeric entries to NA
lab_data_dt_clean <- lab_data_dt_clean %>%
  mutate(num_result = as.numeric(num_result))

# Check for entries that were converted to NA
na_num_result_entries_after <- lab_data_dt_clean %>%
  filter(is.na(num_result))

# Print non-numeric entries that were set to NA after conversion
if (nrow(na_num_result_entries_after) > 0) {
  print("Non-numeric entries found in num_result and converted to NA:")
  print(na_num_result_entries_after)
}

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

#-------------------------------------------------------------------------------
# Remove values incompatible with life

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
  filter_incompatible_values(., "NA", limits$Natrium[1], limits$Natrium[2]) %>%  # Sodium
  filter_incompatible_values(., "KA", limits$Kalium[1], limits$Kalium[2]) %>%    # Potassium
  filter_incompatible_values(., "CL", limits$Chloride[1], limits$Chloride[2]) %>%# Chloride
  filter_incompatible_values(., "pH", limits$pH[1], limits$pH[2]) %>%            # Arterial pH
  filter_incompatible_values(., "pHa", limits$pH[1], limits$pH[2]) %>%           # Venous pH
  filter_incompatible_values(., "pHv", limits$pH[1], limits$pH[2])               # Unspecified pH

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
  
  # Optionally, remove remaining duplicates
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