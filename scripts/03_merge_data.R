#===============================================================================
# Script: 03_merge_data.R
# Purpose: This script merges laboratory and clinical datasets, aggregates lab test results
#          to mean, min, max, median, and first value for each test, transforms the data into a wide format,
#          and categorizes patients based on pseudo_patient_id. Tests with more than 80% missing data are 
#          removed based on the Pareto principle (80/20 rule) to focus on the most reliable tests.
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
library(ggplot2)
setDTthreads(8)  # Set threads for data.table (optional for parallel processing)

# Parameters
lab_data_path <- "./output_files/cleaned_lab_data.csv"
cleaned_data_path <- "./output_files/cleaned_clinical_data.csv"
merged_data_path <- "./output_files/merged_data.csv"

#===============================================================================
# 1. Load the input data
#===============================================================================
lab_data <- read.csv(lab_data_path)
clinical_data <- read.csv(cleaned_data_path)

# View first few rows and summaries
head(lab_data)
head(clinical_data)

# Print number of unique case IDs and patient IDs in both datasets
cat("Unique case IDs in lab data:", uniqueN(lab_data$pseudo_case_id), "\n")
cat("Unique patient IDs in lab data:", uniqueN(lab_data$pseudo_patient_id), "\n")
cat("Unique case IDs in clinical data:", uniqueN(clinical_data$pseudo_case_id), "\n")
cat("Unique patient IDs in clinical data:", uniqueN(clinical_data$pseudo_patient_id), "\n")

# Count unique test_abbr in lab dataset
cat("Unique test abbreviations in lab data:", uniqueN(lab_data$test_abbr), "\n")

# Convert lab and clinical data to data.tables
lab_data_dt <- as.data.table(lab_data)
clinical_data_dt <- as.data.table(clinical_data)

#===============================================================================
# 2. Check for duplicates before merging
#===============================================================================
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

rm(lab_duplicates)
rm(clinical_duplicates)

#===============================================================================
# 3. Check for Inconsistent Test Names and Abbreviations
#===============================================================================
# Identify inconsistencies in test_abbr and test_name
# Group by test_abbr to find multiple test_name values, and sort alphabetically
abbr_to_name_inconsistency <- lab_data_dt[, 
                                          .(unique_test_names = uniqueN(test_name), 
                                            test_names = paste(unique(test_name), collapse = ", ")), 
                                          by = test_abbr][unique_test_names > 1][order(test_abbr)]

# Group by test_name to find multiple test_abbr values, and sort alphabetically
name_to_abbr_inconsistency <- lab_data_dt[, 
                                          .(unique_test_abbr = uniqueN(test_abbr), 
                                            test_abbrs = paste(unique(test_abbr), collapse = ", ")), 
                                          by = test_name][unique_test_abbr > 1][order(test_name)]

View(name_to_abbr_inconsistency)
View(abbr_to_name_inconsistency)

rm(name_to_abbr_inconsistency, abbr_to_name_inconsistency)
gc()

#===============================================================================
# 4. Filter out tests with more than 80% missing data (Pareto rule)
#===============================================================================
# Filter to include only rows with a valid `num_result` value
filtered_lab_data <- lab_data_dt[!is.na(num_result)]

# Get the total number of unique cases
total_cases <- filtered_lab_data[, .N, by = pseudo_case_id][, .N]

# Count distinct cases for each `test_abbr`
test_case_counts <- filtered_lab_data %>%
  distinct(pseudo_case_id, test_abbr) %>% # Get unique combinations
  count(test_abbr, name = "case_count")    # Count cases for each test_abbr

# Calculate the percentage of cases each test_abbr appears in
test_case_counts <- test_case_counts %>%
  mutate(percentage_cases = (case_count / total_cases) * 100) %>%
  arrange(desc(percentage_cases))          # Sort by percentage 

# Filter for tests measured in at least 20% of cases
filtered_tests <- test_case_counts %>%
  filter(percentage_cases >= 20)

# Print the results
print(filtered_tests)
str(filtered_tests)

# PLOT
# Bar plot of percentage of cases for each test_abbr
ggplot(filtered_tests, aes(x = reorder(test_abbr, percentage_cases), y = percentage_cases)) +
  geom_bar(stat = "identity", fill = "skyblue", alpha = 0.7) +
  labs(title = "Percentage of Cases for Each Test", x = "Test Abbreviation", y = "Percentage of Cases") +
  coord_flip() +  # Flip the axes for readability
  theme_minimal()

# Get the test_abbr values that meet the 20% criterion
tests_to_keep <- filtered_tests$test_abbr

# Filter the original lab dataset to keep only the tests in the tests_to_keep list
filtered_lab_data_pareto <- filtered_lab_data[test_abbr %in% tests_to_keep]

# Check the result
print(head(filtered_lab_data_pareto))
str(filtered_lab_data_pareto)

#===============================================================================
# 5. Re-examining Inconsistent Test Names and Abbreviations in Filtered Data
#===============================================================================
# Check the test names associated with each test_abbr
test_name_by_abbr <- filtered_lab_data_pareto[, 
                                              .(unique_test_names = paste(unique(test_name), collapse = ", "),  # List all unique test names
                                                test_name_counts = paste(table(test_name), collapse = ", ")),  # Count occurrences of each test_name
                                              by = test_abbr]

# Filter rows where there are more than 1 unique test_name per test_abbr
test_name_by_abbr <- test_name_by_abbr[, num_test_names := lengths(strsplit(unique_test_names, ", "))] # Count unique test names

# View the results of test names by test_abbr
cat("\nTest names by test_abbr:\n")
print(test_name_by_abbr[order(test_name_by_abbr$test_abbr), ])

# Only print the test abbreviations associated with more than one unique test name
cat("\nTest abbreviations associated with multiple test names:\n")
print(test_name_by_abbr[num_test_names > 1][order(test_abbr), ])

rm(test_name_by_abbr)

#===============================================================================
# 6. Check 'Unit' Consistency Across Test Abbreviations 
#===============================================================================
# Group by test_abbr and summarize the unique units and their frequencies
unit_consistency <- filtered_lab_data_pareto[, 
                                             .(unique_units = uniqueN(unit), 
                                               units = paste(names(table(unit)), collapse = ", "),  # Use names of the table (i.e., the unique units)
                                               unit_counts = paste(as.numeric(table(unit)), collapse = ", ")),  # Use counts from the table
                                             by = test_abbr]

# Save unit_consistency
write.csv(unit_consistency, "./output_files/pareto_tests_unit.csv", row.names = FALSE)

# Identify any test_abbr with more than one unique unit
inconsistent_units <- unit_consistency[unique_units > 1]

# Print inconsistent units and their counts for inspection
if (nrow(inconsistent_units) > 0) {
  cat("Inconsistent units found for the following test abbreviations:\n")
  print(inconsistent_units)
} else {
  cat("All test abbreviations have consistent units.\n")
}

# Clean and standardize unit labels to avoid inconsistencies
filtered_lab_data_pareto[, unit := trimws(tolower(unit))]  # Convert to lowercase and remove any extra spaces

# Set "None" as unit for pH tests
filtered_lab_data_pareto[test_abbr == "pH", unit := "none"]

# Filter out EPIGFR and pH from unknown_analysis before furthere analysis
inconsistent_units <- inconsistent_units[!(test_abbr %in% c("EPIGFR", "pH"))]

# Focus on rows from the data set where unit is inconsistent or labeled "Unknown"
unknown_analysis <- filtered_lab_data_pareto[test_abbr %in% inconsistent_units$test_abbr]

# Add a new column to categorize the unit as either 'Known' or 'Unknown'
unknown_analysis[, unit_type := ifelse(unit == "unknown", "Unknown", "Known")]

# Plot boxplot of values for tests with inconsistent units
ggplot(unknown_analysis, aes(x = unit_type, y = num_result, fill = unit_type)) +
  geom_boxplot(alpha = 0.7, 
               outlier.shape = 16,       # Choose the shape of the dots (16 = solid circle)
               outlier.color = "darkgrey", # Set the color of the outliers
               outlier.size = 2,         # Adjust the size of the outliers
               outlier.alpha = 0.4) +    # Adjust the transparency of the outliers
  facet_wrap(~ test_abbr, scales = "free") +  # Separate plots for each test_abbr
  labs(title = "Boxplot: Distribution of Values by Known vs Unknown Units for Inconsistent Units Across Test Abbreviations",
       x = "Unit Type", y = "Value") +
  theme_minimal() +
  scale_fill_manual(values = c("Unknown" = "skyblue", "Known" = "coral"))  # Set colors for box fill

# Adjusted Boxplot (Using Log Scale)
ggplot(unknown_analysis, aes(x = unit_type, y = num_result, fill = unit_type)) +
  geom_boxplot(alpha = 0.7, 
               outlier.shape = 16,
               outlier.color = "darkgrey", 
               outlier.size = 2) +
  scale_y_log10() +  # Log scale for y-axis
  facet_wrap(~ test_abbr, scales = "free") +
  labs(title = "Boxplot: Distribution of Values by Known vs Unknown Units for Inconsistent Units Across Test Abbreviations",
       x = "Unit Type", y = "Value") +
  theme_minimal() +
  scale_fill_manual(values = c("Unknown" = "skyblue", "Known" = "coral"))

# Calculate summary statistics for inconsistent or unknown units
unknown_summary <- unknown_analysis[, .(
  mean_value = mean(num_result, na.rm = TRUE),
  median_value = median(num_result, na.rm = TRUE),
  sd_value = sd(num_result, na.rm = TRUE),
  min_value = min(num_result, na.rm = TRUE),
  max_value = max(num_result, na.rm = TRUE),
  n = .N  # Count the number of entries
), by = .(test_abbr, unit_type)]

# Output summary statistics
cat("\nSummary Statistics for Inconsistent or Unknown Units:\n")
print(unknown_summary %>% arrange(test_abbr))  # Sort by test_abbr to group them together)

#-------------
# Group by test_abbr and summarize the unique units and their frequencies after correction (epgfr and ph)
unit_consistency <- filtered_lab_data_pareto[, 
                                             .(unique_units = uniqueN(unit), 
                                               units = paste(names(table(unit)), collapse = ", "),  # Use names of the table (i.e., the unique units)
                                               unit_counts = paste(as.numeric(table(unit)), collapse = ", ")),  # Use counts from the table
                                             by = test_abbr]

# Save unit_consistency
write.csv(unit_consistency, "./output_files/pareto_tests_unit.csv", row.names = FALSE)
#-------------

rm(unit_consistency, unknown_analysis, unknown_summary)
gc()

#===============================================================================
# 7. Aggregate lab data (mean, min, max, median, first_value)
#===============================================================================
# Group by pseudo_patient_id, pseudo_case_id, and test_abbr to calculate the statistics
aggregated_lab_data <- filtered_lab_data_pareto[, 
                                                .(mean = mean(num_result, na.rm = TRUE),  # Mean of num_result
                                                  median_value = median(num_result, na.rm = TRUE),  # Median value
                                                  min_value = min(num_result, na.rm = TRUE),  # Minimum value
                                                  max_value = max(num_result, na.rm = TRUE),  # Maximum value
                                                  first_value = first(num_result)),  # First value using first() function
                                                by = .(pseudo_patient_id, pseudo_case_id, test_abbr)
]

# Melt the data to reshape it into long format
melted_lab_data <- melt(aggregated_lab_data, 
                        id.vars = c("pseudo_patient_id", "pseudo_case_id", "test_abbr"), 
                        measure.vars = c("mean", "median_value", "min_value", "max_value", "first_value"), 
                        variable.name = "statistic", value.name = "value")

#===============================================================================
# 8. Transform the data to wide format
#===============================================================================
# Now use dcast to convert back to wide format
wide_aggregated_lab_data <- dcast(melted_lab_data, 
                                  pseudo_patient_id + pseudo_case_id ~ test_abbr + statistic, 
                                  value.var = "value")

# Check the wide format data
cat("\nWide format data (first few rows):\n")
print(head(wide_aggregated_lab_data))
str(wide_aggregated_lab_data)

gc()

#===============================================================================
# 9. Merge datasets
#===============================================================================
# Merge clinical data with the aggregated lab data (keeping important patient info)
merged_data <- clinical_data_dt %>%
  select(pseudo_patient_id, pseudo_case_id, discharge_type, sex, age, length_of_stay_days, principal_diagnosis) %>%
  inner_join(wide_aggregated_lab_data, by = c("pseudo_patient_id", "pseudo_case_id"))

# Check the merged data
cat("\nMerged data (first few rows):\n")
print(head(merged_data))

str(merged_data)

#===============================================================================
# 10. Save the merged dataset
#===============================================================================
write.csv(merged_data, merged_data_path, row.names = FALSE)
cat("\nMerged data saved to:", merged_data_path, "\n")

#===============================================================================
# Summary of Merging Steps:
# - Load and inspect the input datasets (lab data, clinical data).
# - Filter out numeric results from the lab data.
# - Apply Pareto rule: Filter out tests with more than 80% missing data.
# - Checks for inconsistencies in test abbreviations, names, and units.
# - Summarizes and outputs statistics for inconsistent and unknown units.
# - Aggregate lab data (mean, min, max, median, first_value).
# - Convert the aggregated data into wide format (one row per patient/case, with columns for each test).
# - Merge the aggregated lab data with clinical data.
# - Save the merged dataset for further analysis.
#===============================================================================
