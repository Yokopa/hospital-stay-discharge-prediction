#===============================================================================
# Script: 04_merged_DE.R
# Purpose: Analyze lab test data, compute correlations, and visualize relationships 
#          between lab tests, length of stay (LOS), and other demographic factors.
# Data Source: Insel Data Science Center (IDSC)
# Author: Anna Scarpellini Pancrazi
# Date: 2024
# Input File: lab_data.csv (cleaned lab data with translated English column names)
# Output Files:
#   - Correlation matrix plot (correlation_matrix.png)
#   - Correlation with LOS plot (correlation_with_los.png)
#   - LOS by Diagnosis and Discharge Type heatmap (LOS_heatmap.png)
#   - Length of Stay by Age Group boxplot (LOS_by_age_group.png)
#   - Faceted Scatterplot by Age Group (LOS_by_diagnosis_age_group.png)
#===============================================================================

# Packages
library(data.table)      
library(ggplot2)        
library(ggcorrplot)      
library(dplyr)           
library(viridis)        

# Load the data
data <- read.csv("./output_files/merged_data.csv") 

#===============================================================================
# Basic Data Exploration (Clinical data)
#===============================================================================
############################
# Age and Sex Distribution #
############################

# Basic summary for age
summary(data$age)

# Age distribution histogram
ggplot(data, aes(x = age)) +
  geom_histogram(binwidth = 1, fill = "skyblue", color = "white", alpha = 0.7) +
  theme_minimal() + # Removes the grey background
  labs(title = "Age Distribution", x = "Age", y = "Frequency") +
  theme(
    axis.title.x = element_text(size = 16), # Adjusts the size of x-axis title
    axis.title.y = element_text(size = 16), # Adjusts the size of y-axis title
    axis.text.x = element_text(size = 14),  # Adjusts the size of x-axis text
    axis.text.y = element_text(size = 14)   # Adjusts the size of y-axis text
  )
# ------------------------------------------------------------------------------

# Frequency distribution of sex with percentages
sex_distribution <- data %>%
  count(sex) %>%
  mutate(percentage = n / sum(n) * 100)

# View the result
print(sex_distribution)

# Sex distribution bar plot with percentages
ggplot(sex_distribution, aes(x = sex, y = n)) +
  geom_bar(stat = "identity", fill = "skyblue", color = "black", alpha = 0.7) +
  geom_text(aes(label = paste0(round(percentage, 1), "%")), vjust = -0.5) +
  labs(title = "Sex Distribution", x = "Sex", y = "Count") +
  theme_minimal()

# ------------------------------------------------------------------------------

############################
#    Diagnosis Analysis    #
############################

# Count the frequency of each principal diagnosis
diagnosis_counts <- data %>%
  count(principal_diagnosis) %>%
  arrange(desc(n))

# Function to extract the top N% of diagnoses
get_top_percent <- function(data, percent) {
  cutoff_index <- ceiling(nrow(data) * (percent / 100)) # Calculate the index cutoff
  data[1:cutoff_index, ]  # Return the top rows up to the cutoff
}

# Apply the function for the top 25% of diagnoses
top_25_percent <- get_top_percent(diagnosis_counts, 25)

# Plot the top 25% of diagnoses
ggplot(top_25_percent, aes(x = reorder(principal_diagnosis, -n), y = n)) +
  geom_bar(stat = "identity", fill = "skyblue", color = "black", alpha = 0.7) +
  labs(
    title = "Most Frequent Diagnoses (Top 25%)",
    x = "Principal Diagnosis",
    y = "Count"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 6)) +
  theme_minimal()

rm(top_25_percent)
gc()

#----------------

# Plot top diagnoses
plot_top_diagnoses <- function(data, top_n) {
  top_diagnoses <- data %>%
    head(top_n)
  ggplot(top_diagnoses, aes(x = reorder(principal_diagnosis, -n), y = n)) +
    geom_bar(stat = "identity", fill = "skyblue", alpha = 0.7) +
    labs(
      title = paste("Top", top_n, "Most Frequent Diagnoses"),
      x = "Principal Diagnosis",
      y = "Count"
    ) +
    theme_minimal() + # Removes the grey background
    theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  theme(
    axis.title.x = element_text(size = 16), # Adjusts the size of x-axis title
    axis.title.y = element_text(size = 16), # Adjusts the size of y-axis title
    axis.text.x = element_text(size = 14),  # Adjusts the size of x-axis text
    axis.text.y = element_text(size = 14)   # Adjusts the size of y-axis text
  )
}

# Use the function for top 100, top 50, top 20
plot_top_diagnoses(diagnosis_counts, 100)
plot_top_diagnoses(diagnosis_counts, 50)
plot_top_diagnoses(diagnosis_counts, 20)

cat("Top 20 Most Frequent Diagnoses:\n")
print(diagnosis_counts %>% head(20))

#----------------

# Simplify and categorize diagnoses
data$diagnosis_category <- case_when(
  grepl("^A|^B", data$principal_diagnosis) ~ "Infectious diseases",
  grepl("^C|^D[0-4]", data$principal_diagnosis) ~ "Neoplasms",
  grepl("^D[5-8]", data$principal_diagnosis) ~ "Blood and immune disorders",
  grepl("^E", data$principal_diagnosis) ~ "Endocrine and metabolic diseases",
  grepl("^F", data$principal_diagnosis) ~ "Mental and behavioral disorders",
  grepl("^G", data$principal_diagnosis) ~ "Nervous system diseases",
  grepl("^H[0-5]", data$principal_diagnosis) ~ "Eye diseases",
  grepl("^H[6-9]", data$principal_diagnosis) ~ "Ear diseases",
  grepl("^I", data$principal_diagnosis) ~ "Circulatory diseases",
  grepl("^J", data$principal_diagnosis) ~ "Respiratory diseases",
  grepl("^K", data$principal_diagnosis) ~ "Digestive diseases",
  grepl("^L", data$principal_diagnosis) ~ "Skin diseases",
  grepl("^M", data$principal_diagnosis) ~ "Musculoskeletal diseases",
  grepl("^N", data$principal_diagnosis) ~ "Genitourinary diseases",
  grepl("^O", data$principal_diagnosis) ~ "Pregnancy-related conditions",
  grepl("^P", data$principal_diagnosis) ~ "Perinatal conditions",
  grepl("^Q", data$principal_diagnosis) ~ "Congenital conditions",
  grepl("^R", data$principal_diagnosis) ~ "Symptoms and ill-defined conditions",
  grepl("^S|^T", data$principal_diagnosis) ~ "Injuries and poisoning",
  grepl("^V|^W|^X|^Y", data$principal_diagnosis) ~ "External causes",
  grepl("^Z", data$principal_diagnosis) ~ "Health service factors",
  grepl("^U", data$principal_diagnosis) ~ "Special purposes",
  is.na(data$principal_diagnosis) ~ "Unknown"
)

# Check the distribution of the new categories
table(data$diagnosis_category)
print(sort(table(data$diagnosis_category), decreasing = TRUE))

# Plot the distribution of diagnosis categories
# Convert the diagnosis table into a data frame
diagnosis_counts <- as.data.frame(table(data$diagnosis_category))
colnames(diagnosis_counts) <- c("Category", "Count")

# Create a horizontal bar chart
ggplot(diagnosis_counts, aes(x = reorder(Category, Count), y = Count)) +
  geom_bar(stat = "identity", fill = "skyblue", alpha = 0.7) +
  coord_flip() +  # Flip coordinates for horizontal bars
  labs(
    title = "Distribution of Diagnosis Categories",
    x = "Diagnosis Category",
    y = "Number of Cases"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(size = 14),
    axis.text.y = element_text(size = 14),
    axis.title.x = element_text(size = 16), # Adjusts the size of x-axis title
    axis.title.y = element_text(size = 16), # Adjusts the size of y-axis title
    plot.title = element_text(hjust = 0.5, size = 14)
  )

rm(diagnosis_counts)
gc()

# ------------------------------------------------------------------------------

############################
#      Length of Stay      #
############################
setDT(data)

# Distribution (Boxplot and Histogram)
summary(data$length_of_stay_days)

## Length of stay boxplot
ggplot(data, aes(y = length_of_stay_days), aes(x="")) +
  geom_boxplot(color = "skyblue", alpha = 0.7) +
  labs(title = "Length of Stay Boxplot", y = "Length of Stay (Days)") +
  theme_minimal() +
  theme(
    axis.title.y = element_text(size = 16),
    axis.text.y = element_text(size = 14),
    axis.text.x = element_blank(),  # Remove x-axis numbers
    axis.ticks.x = element_blank()  # Remove x-axis ticks
  )

# Scatter plot with case_id on x-axis and length_of_stay_days (LOS) on y-axis
ggplot(data, aes(x = pseudo_case_id, y = length_of_stay_days)) +
  geom_point(color = "skyblue", alpha = 0.6, shape = 16) +  # Scatter plot with points
  labs(title = "Scatter Plot of Length of Stay by Case ID", x = "Case ID", y = "Length of Stay (Days)") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
    axis.title.x = element_text(size = 16),
    axis.title.y = element_text(size = 16),
    axis.text.x = element_blank(),   # Remove x-axis labels
    axis.ticks.x = element_blank(),  # Remove x-axis ticks
  )

# Calculate the IQR for length_of_stay_days
Q1 <- quantile(data$length_of_stay_days, 0.25, na.rm = TRUE)
Q3 <- quantile(data$length_of_stay_days, 0.75, na.rm = TRUE)
IQR <- Q3 - Q1

# Define upper bound for outliers
upper_bound <- Q3 + 1.5 * IQR

# Identify extraordinarily long stays (outliers above the upper bound) in descending order length
extraordinarily_long_stays_IQR <- data %>% filter(length_of_stay_days > upper_bound) %>% 
  arrange(desc(length_of_stay_days))

# Print outliers based on IQR
cat("The upper bound for outliers in length of stay is:", upper_bound, "\n")
cat("Number of extraordinarily long stays identified using IQR:", nrow(extraordinarily_long_stays_IQR), "\n")

# Print out the extraordinarily long stays, in descending orderlength
print(extraordinarily_long_stays_IQR[,1:7])
nrow(extraordinarily_long_stays_IQR)

# Define the lower bound for outliers
lower_bound <- Q1 - 1.5 * IQR
cat("The lower bound for outliers in length of stay is:", lower_bound, "\n")

# Select the first 20 entries with the shortest length of stay
lower_entries <- data %>%
  arrange(length_of_stay_days) %>%
  slice_head(n = 20)

print(lower_entries[,1:7])

#----------------

# Calculate the 99th percentile
percentile_99 <- quantile(data$length_of_stay_days, 0.99, na.rm = TRUE)

# Identify stays above the 99th percentile
extraordinarily_long_stays_99 <- data %>% filter(length_of_stay_days > percentile_99) %>% 
  arrange(desc(length_of_stay_days))

print(percentile_99)

# Print out the extraordinarily long stays
print(extraordinarily_long_stays_99[,1:7])
nrow(extraordinarily_long_stays_99)

# Sort the dataset by Length of Stay (LOS) in descending order and select the first 10 rows
longest_los <- data[order(-data$length_of_stay_days), ][1:10, ]

# View the longest LOS cases
print(longest_los[,1:7])

#----------------

# Filter patients with length_of_stay_days = 0
zero_length_stay <- data %>% filter(length_of_stay_days == 0)

# Check the most frequent diagnosis codes (assuming 'diagnosis_code' is the column name)
frequent_diagnosis_codes_zero <- zero_length_stay %>%
  group_by(principal_diagnosis) %>%
  tally() %>%
  arrange(desc(n))

# Print out the most frequent diagnosis codes
print(frequent_diagnosis_codes_zero)

# Check the most frequent diagnosis categories (assuming 'diagnosis_category' is the column name)
frequent_diagnosis_categories_zero <- zero_length_stay %>%
  group_by(diagnosis_category) %>%
  tally() %>%
  arrange(desc(n))

# Print out the most frequent diagnosis categories
print(frequent_diagnosis_categories_zero)

#----------------

# Plotting Length of Stay Distribution
# These histograms will show the frequency of different lengths of stay.

# Define x-axis limits for different range analyses
x_limits_list <- list(c(0, 500), c(0, 300), c(0, 100), c(0, 50), c(0, 10), c(70, 500), c(100, 500))

# Loop through each x-axis limit and create a histogram
for (x_lim in x_limits_list) {
  plot <- ggplot(data, aes(x = length_of_stay_days)) +
      geom_histogram(binwidth = 1, fill = "skyblue", color = "black", alpha = 0.7) +
      labs(title = paste("Length of Stay Distribution (Range: ", x_lim[1], "-", x_lim[2], " days)", sep = ""),
          x = "Length of Stay (days)", 
          y = "Frequency") +
      xlim(x_lim[1], x_lim[2]) + 
      theme_minimal()
    # Explicitly print the plot
    print(plot)
}

gc()

#----------------

# Set the scipen option to avoid scientific notation
options(scipen = 999)

# Define thresholds for monthly, weekly, and daily ranges
days_1_month <- 30
days_2_months <- 60
days_3_months <- 90

# Helper function to calculate and format the results
calculate_summary <- function(data, breaks) {
  data %>%
    mutate(range = cut(length_of_stay_days, breaks = breaks, include.lowest = TRUE, right = FALSE)) %>%
    group_by(range) %>%
    summarize(
      cases = n(),
      percentage = round((cases / nrow(data)) * 100, 1)
    ) %>%
    ungroup()
}

# Monthly ranges: 0-1 month, 1-2 months, 2-3 months, 3+ months
monthly_breaks <- c(0, days_1_month, days_2_months, days_3_months, Inf)
monthly_results <- calculate_summary(data, monthly_breaks)

# Print the monthly results
cat("### Monthly Breakdown\n")
knitr::kable(monthly_results, format = "markdown", align = "c", caption = "Length of stay analysis with monthly ranges")

# Monthly Breakdown Bar Chart
ggplot(monthly_results, aes(x = range, y = cases, fill = range)) +
  geom_bar(stat = "identity", fill = "skyblue", alpha=0.7) +
  labs(title = "Distribution of Cases by Length of Stay (Monthly Ranges)", x = "Month Range", y = "Number of Cases") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 0, hjust = 1), legend.position = "none") +
  scale_x_discrete(labels = c("0-1 month", "1-2 months", "2-3 months", "3+ months"))

# Weekly breakdown for the first month: 0-1 week, 1-2 weeks, 2-3 weeks, 3-4 weeks
weekly_breaks <- c(seq(0, 28, by = 7), Inf)  # Weekly intervals (0-7, 7-14, 14-21, etc., and 28+)
weekly_results <- calculate_summary(data, weekly_breaks)

# Print the weekly results
cat("\n### Weekly Breakdown\n")
knitr::kable(weekly_results, format = "markdown", align = "c", caption = "Length of stay analysis with weekly ranges")

# Weekly Breakdown Bar Chart
ggplot(weekly_results, aes(x = range, y = cases, fill = range)) +
  geom_bar(stat = "identity", fill = "skyblue", alpha=0.7) +
  labs(title = "Distribution of Cases by Length of Stay (Weekly Ranges)", x = "Week Range", y = "Number of Cases") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 0, hjust = 1), legend.position = "none") +
  scale_x_discrete(labels = c("0-1 week", "1-2 weeks", "2-3 weeks", "3-4 weeks", "4+ weeks"))+
  theme(
    axis.title.x = element_text(size = 16), # Adjusts the size of x-axis title
    axis.title.y = element_text(size = 16), # Adjusts the size of y-axis title
    axis.text.x = element_text(size = 14),  # Adjusts the size of x-axis text
    axis.text.y = element_text(size = 14)   # Adjusts the size of y-axis text
  )

# Helper function to calculate and format the results for the first 14 days
calculate_exact_summary_14_days <- function(data) {
  data %>%
    filter(length_of_stay_days <= 14) %>%  # Filter for first 14 days
    group_by(length_of_stay_days) %>%
    summarize(
      cases = n(),
      percentage = round((cases / nrow(data)) * 100, 1)
    ) %>%
    ungroup()
}

# Calculate the exact summary for length of stay days <= 14
exact_results_14_days <- calculate_exact_summary_14_days(data)

# View the results
print(exact_results_14_days)

# Plot the distribution of Length of Stay for the first 14 days
ggplot(exact_results_14_days, aes(x = as.factor(length_of_stay_days), y = cases)) +
  geom_bar(stat = "identity", fill = "skyblue", alpha = 0.7) +
  labs(title = "Distribution of Length of Stay (0-14 Days)", 
       x = "Length of Stay (Days)", 
       y = "Number of Cases") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 0, hjust = 1))+
  theme(
    axis.title.x = element_text(size = 16), # Adjusts the size of x-axis title
    axis.title.y = element_text(size = 16), # Adjusts the size of y-axis title
    axis.text.x = element_text(size = 14),  # Adjusts the size of x-axis text
    axis.text.y = element_text(size = 14)   # Adjusts the size of y-axis text
  )

#----------------

# Filter the data for length_of_stay_days = 0 and calculate the ranking of discharge types
discharge_ranking_0_days <- data %>%
  filter(length_of_stay_days == 0) %>%
  group_by(discharge_type) %>%
  summarize(cases = n()) %>%
  arrange(desc(cases))  # Order by most frequent discharge type

# View the ranking
print(discharge_ranking_0_days)

# Filter the data for length_of_stay_days = 0 and calculate the ranking of diagnoses
diagnosis_ranking_0_days <- data %>%
  filter(length_of_stay_days == 0) %>%
  group_by(principal_diagnosis) %>%  # Replace 'diagnosis' with the actual column name for diagnosis
  summarize(cases = n()) %>%
  arrange(desc(cases))  # Order by most frequent diagnosis

# View the ranking
print(diagnosis_ranking_0_days)

# ------------------------------------------------------------------------------

############################
#      Discharge Type      #
############################
# Proportions of Different Discharge Types

# Calculate the count and percentage of each discharge type
discharge_summary <- data %>%
  group_by(discharge_type) %>%
  summarize(
    count = n(),
    percentage = (n() / nrow(data)) * 100
  ) %>%
  arrange(desc(count))  # Sort by count in descending order

# Print the discharge summary
print(discharge_summary)

# Plot the bar chart
ggplot(discharge_summary, aes(x = reorder(discharge_type, -count), y = count)) +
  geom_bar(stat = "identity", fill = "skyblue", alpha = 0.7) +
  labs(title = "Discharge Type Proportions", x = "Discharge Type", y = "Count") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  geom_text(aes(label = paste0(round(percentage, 1), "%")), vjust = -0.5)  # Add percentages on top of bars

# Plot the bar chart with counts on top
ggplot(discharge_summary, aes(x = reorder(discharge_type, -count), y = percentage)) +
  geom_bar(stat = "identity", fill = "skyblue", alpha = 0.7) +
  labs(title = "Discharge Type Proportions", x = "Discharge Type", y = "Percentage") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  geom_text(aes(label = count), vjust = -0.5, size = 5) + # Adjusted size for text labels
  theme(
    axis.title.x = element_text(size = 16), # Adjusts the size of x-axis title
    axis.title.y = element_text(size = 16), # Adjusts the size of y-axis title
    axis.text.x = element_text(size = 14),  # Adjusts the size of x-axis text
    axis.text.y = element_text(size = 14)   # Adjusts the size of y-axis text
  )

#----------------

# Analysis of the deceased patients

# Filter the data for deceased patients and summarize key statistics
deceased_data <- data %>%
  filter(discharge_type == "DECEASED")

# Summary of deceased patients by key characteristics
deceased_summary <- deceased_data %>%
  summarize(
    total_deceased = n(),
    average_age = mean(age, na.rm = TRUE),
    length_of_stay_mean = mean(length_of_stay_days, na.rm = TRUE),
    most_common_diagnosis = names(sort(table(principal_diagnosis), decreasing = TRUE))[1]
  )

# Calculate percentages for the top 20 diagnoses
diagnosis_deceased_summary <- deceased_data %>%
  group_by(principal_diagnosis) %>%
  summarize(count = n()) %>%
  arrange(desc(count)) %>%
  head(20)  # Get the top 20 most common diagnoses

# Add percentage column
diagnosis_deceased_summary <- diagnosis_deceased_summary %>%
  mutate(percentage = round((count / sum(count)) * 100, 1))

# Print the summary with percentages
print(diagnosis_deceased_summary)

# Plot the top 20 most common diagnoses among deceased patients
ggplot(diagnosis_deceased_summary, aes(x = reorder(principal_diagnosis, -count), y = count)) +
  geom_bar(stat = "identity", fill = "skyblue", alpha = 0.7) +
  labs(title = "Top 20 Most Common Diagnoses for Deceased Patients", x = "Diagnosis", y = "Count") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 0, hjust = 1)) +
  geom_text(aes(label = paste0(percentage, "%")), vjust = -0.5, size = 5) +
  theme(
    axis.title.x = element_text(size = 16), # Adjusts the size of x-axis title
    axis.title.y = element_text(size = 16), # Adjusts the size of y-axis title
    axis.text.x = element_text(size = 14),  # Adjusts the size of x-axis text
    axis.text.y = element_text(size = 14)   # Adjusts the size of y-axis text
  )

rm(deceased_summary)
gc()

#===============================================================================
# Basic Data Exploration (Lab data)
#===============================================================================

# Test Frequency - Most Frequently Administered Lab Tests
# Subset the data from column 8 onwards, excluding the last column ("patient_cluster")
lab_data <- data[, 8:(ncol(data)-1)]
setDT(lab_data) # Convert data frame to data.table for efficient processing

# Count the number of non-NA values for each lab parameter
lab_frequencies <- colSums(!is.na(lab_data))

# Sort the frequencies in descending order to identify the most frequently administered tests
lab_frequencies_sorted <- sort(lab_frequencies, decreasing = TRUE)

# View in descending order from the most frequently measured lab tests
print(lab_frequencies_sorted)

# Count total number of columns (lab tests) being analyzed
length(lab_frequencies_sorted)

# ------------------------------------------------------------------------------

##################################################################
# Extract test names and determine preferred units for each test #
##################################################################

# Package used: stringr, data.table
# Extract test names from column prefixes (e.g., ALAT from ALAT_mean)
column_groups <- str_extract(names(lab_frequencies_sorted), "^[^_]+") # Extract part before "_"
unique_tests <- unique(column_groups)  # Get unique test names

# Load table with tests and corresponding entered units from previous script 03_merge_data
unit_table <- read.csv("./output_files/pareto_tests_unit.csv")
# Convert unit_table to data.table
unit_table <- as.data.table(unit_table)

# Filter unit_table to only include tests in unique_tests
filtered_units <- unit_table[test_abbr %in% unique_tests]

# For each test_abbr, select the preferred unit (handle "unknown" units)
final_units <- filtered_units[, .(
  preferred_unit = ifelse("unknown" %in% units & unique_units > 1, 
                          setdiff(units, "unknown"), 
                          units)
), by = test_abbr]

# Handle cases with multiple preferred units and combine them as a string
final_units[, preferred_unit := sapply(preferred_unit, function(x) paste(unique(x), collapse = ", "))]

# Order the final_units data.table by unique_tests list for consistency
final_units <- final_units[order(match(test_abbr, unique_tests))]

# View the result to verify preferred units for each test
print(final_units)

# ------------------------------------------------------------------------------

###################################################################
# Compute lab data results statistics and plot their distribution #
###################################################################

# Customizable: Change the value of `num_tests_to_plot` to limit the number of tests plotted.
# Here it is set to 22, but you can adjust as needed (e.g., change 22 to any number <= 70).
num_tests_to_plot <- 22  # Change this value to adjust how many tests to plot

# Select the "Blues" palette and remove the lightest color
custom_palette <- brewer.pal(9, "Blues")[-1]

unique_tests_to_plot <- unique_tests[1:num_tests_to_plot]  # Limit the number of unique tests to plot

# Loop over the unique tests to create plots for each test
for (i in seq_along(unique_tests_to_plot)) {
  cat("Starting analysis for ", unique_tests[i], "\n")
  
  # Get the column names for the current test (search for test columns by prefix)
  test_cols <- grep(paste0("^", unique_tests[i], "_"), colnames(lab_data), value = TRUE)
  
  # If no columns found, skip to the next test
  if (length(test_cols) == 0) {
    cat("No data found for test:", unique_tests[i], "\n")
    next
  }
  
  # Use ..test_cols for column selection in data.table
  test_data <- lab_data[, ..test_cols]
  
  # Descriptive statistics: Print summary of data for the current test
  cat("\nDescriptive statistics for ", unique_tests[i], ":\n")
  print(summary(test_data))
  
  # Melt the data for ggplot (turn wide data into long format for plotting)
  test_data_long <- melt(test_data, variable.name = "Statistic", value.name = "Value", na.rm = TRUE)
  
  # Check if melting produced any data (skip if no data to plot)
  if (nrow(test_data_long) == 0) {
    cat("No data to plot for test:", unique_tests[i], "\n")
    next
  }
  
  # Get the preferred unit for the test from the final_units data.table
  unit <- final_units$preferred_unit[i]
  
  # Boxplot for each statistic
  plot1 <- ggplot(test_data_long, aes(x = Statistic, y = Value)) +
    geom_boxplot(fill = "lightblue", outlier.color = "navy", outlier.shape = 16) +
    labs(title = paste("Boxplots of", unique_tests[i]),
         x = paste(unique_tests[i], "Statistic"),
         y = paste(unique_tests[i], "Level (", unit, ")")) +
    theme_minimal()
  print(plot1)
  
  # Density plot with custom color palette
  plot4 <- ggplot(test_data_long, aes(Value, color = Statistic)) +
    geom_density(alpha = 0.8) +
    labs(title = paste("Density Plot of", unique_tests[i], "Statistics"), 
         x = paste(unique_tests[i], "Level (", unit, ")"), 
         y = "Density") +
    scale_color_manual(values = custom_palette) +
    theme_minimal()
  print(plot4)
}

# ------------------------------------------------------------------------------

##############################################################################
# Filter and inspect the most extreme values for the KA (potassium) lab test #
##############################################################################

# Filter the data for rows where KA_first_value > 5.0 mmol/L or KA_first_value < 3.5 mmol/L
# Separate the values into two groups: "KA_first_value > 5.0" and "KA_first_value < 3.5"
# We will then examine the first 7 columns and all columns with "KA" in their name.

# Filter for KA_first_value > 5.0 mmol/L (High Potassium)
high_KA <- data[data$KA_first_value > 5.0, .SD, .SDcols = c(1:7, grep("KA", colnames(data)), ncol(data))]

# Filter for KA_first_value < 3.5 mmol/L (Low Potassium)
low_KA <- data[data$KA_first_value < 3.5, .SD, .SDcols = c(1:7, grep("KA", colnames(data)), ncol(data))]

# Sort the high_KA values in descending order to view the most extreme first
high_KA <- high_KA[order(-high_KA$KA_first_value), ]
# Sort the low_KA values in ascending order to view the most extreme first
low_KA <- low_KA[order(low_KA$KA_first_value), ]

# Total number of rows in the dataset
total_rows <- nrow(data)

# Calculate the number of rows and percentage for high potassium
high_KA_count <- nrow(high_KA)
high_KA_percentage <- (high_KA_count / total_rows) * 100

# Calculate the number of rows and percentage for low potassium
low_KA_count <- nrow(low_KA)
low_KA_percentage <- (low_KA_count / total_rows) * 100

# Print the number of rows and percentages in each filtered group
cat("Number of rows with KA_first_value > 5.0 mmol/L (High Potassium):", high_KA_count, 
    "(", round(high_KA_percentage, 2), "% )\n")
cat("Number of rows with KA_first_value < 3.5 mmol/L (Low Potassium):", low_KA_count, 
    "(", round(low_KA_percentage, 2), "% )\n")

# Print the high potassium rows
cat("\nHigh Potassium Values (KA_first_value > 5.0 mmol/L):\n")
print(high_KA)

# Print the low potassium rows
cat("\nLow Potassium Values (KA_first_value < 3.5 mmol/L):\n")
print(low_KA)

#----------------
# Plot histograms for age distribution in both groups
ggplot(high_KA_sorted, aes(x = age)) +
  geom_histogram(binwidth = 5, fill = "lightblue", color = "black", alpha = 0.7) +
  labs(title = "Age Distribution for High Potassium (KA_first_value > 5.0 mmol/L)", 
       x = "Age", y = "Frequency") +
  theme_minimal()

ggplot(low_KA_sorted, aes(x = age)) +
  geom_histogram(binwidth = 5, fill = "lightcoral", color = "black", alpha = 0.7) +
  labs(title = "Age Distribution for Low Potassium (KA_first_value < 3.5 mmol/L)", 
       x = "Age", y = "Frequency") +
  theme_minimal()
#----------------
ggplot() +
  geom_boxplot(data = high_KA, aes(x = "High Potassium", y = length_of_stay_days, fill = "High Potassium"), 
               color = "black", outlier.color = "navy", outlier.shape = 16) +
  geom_boxplot(data = low_KA, aes(x = "Low Potassium", y = length_of_stay_days, fill = "Low Potassium"), 
               color = "black", outlier.color = "red", outlier.shape = 16) +
  labs(title = "Length of Stay Comparison for High and Low Potassium", 
       x = "", y = "Length of Stay (days)") +
  scale_fill_manual(values = c("High Potassium" = "lightblue", "Low Potassium" = "lightcoral")) +
  theme_minimal()

# Filter and print rows where length_of_stay_days > 500
cat("\nRows with Length of Stay > 500 days:\n")
print(high_KA[high_KA$length_of_stay_days > 500, ])

# Filter and print rows where 200 < length_of_stay_days < 300
cat("\nRows with Length of Stay between 200 and 300 days:\n")
print(low_KA[low_KA$length_of_stay_days > 200 & low_KA$length_of_stay_days < 300, ])
#----------------
# Sex distribution for both high and low potassium groups
ggplot(high_KA, aes(x = sex)) +
  geom_bar(fill = "lightblue", color = "black") +
  labs(title = "Sex Distribution for High Potassium (KA_first_value > 5.0 mmol/L)", 
       x = "Sex", y = "Count") +
  theme_minimal()

ggplot(low_KA, aes(x = sex)) +
  geom_bar(fill = "lightcoral", color = "black") +
  labs(title = "Sex Distribution for Low Potassium (KA_first_value < 3.5 mmol/L)", 
       x = "Sex", y = "Count") +
  theme_minimal()
#----------------
# Add an age group column to the dataset for both high and low potassium groups
high_KA[, age_group := ifelse(age < 65, "<65", "≥65")]
low_KA[, age_group := ifelse(age < 65, "<65", "≥65")]

# Function to summarize diagnosis data by age group and potassium group
summarize_diagnosis <- function(data, diagnosis_column) {
  # Calculate counts for the given diagnosis column grouped by age group
  summary <- data[, .N, by = .(get(diagnosis_column), age_group)]
  
  # Rename column for clarity
  setnames(summary, "N", "count")
  
  # Calculate total count per age group and percentage
  summary[, total_count := sum(count), by = age_group]
  summary[, percentage := (count / total_count) * 100]
  
  # Sort by count and percentage in descending order
  setorder(summary, age_group, -count, -percentage)
  
  return(summary)
}

# Summarize diagnosis categories for high and low potassium groups
high_KA_summary_diag_category <- summarize_diagnosis(high_KA, "diagnosis_category")
low_KA_summary_diag_category <- summarize_diagnosis(low_KA, "diagnosis_category")

# Summarize principal diagnoses for high and low potassium groups
high_KA_summary_principal_diag <- summarize_diagnosis(high_KA, "principal_diagnosis")
low_KA_summary_principal_diag <- summarize_diagnosis(low_KA, "principal_diagnosis")

# Function to print top 5 for each age group within each potassium group
print_top_5 <- function(summary_data, diagnosis_column) {
  # Loop through each age group to print the top 5
  for(age in unique(summary_data$age_group)) {
    cat("\nTop 5", diagnosis_column, "for Age Group:", age, "\n")
    
    # Filter for the current age group and get the top 5 most frequent diagnoses
    top_5 <- summary_data[age_group == age][order(-count)][1:5]
    
    # Print the top 5 rows for the current age group
    print(top_5)
  }
}

# Print top 5 diagnosis categories for both high and low potassium groups
cat("High Potassium - Diagnosis Categories:\n")
print_top_5(high_KA_summary_diag_category, "diagnosis_category")

cat("\nLow Potassium - Diagnosis Categories:\n")
print_top_5(low_KA_summary_diag_category, "diagnosis_category")

# Print top 5 principal diagnoses for both high and low potassium groups
cat("\nHigh Potassium - Principal Diagnoses:\n")
print_top_5(high_KA_summary_principal_diag, "principal_diagnosis")

cat("\nLow Potassium - Principal Diagnoses:\n")
print_top_5(low_KA_summary_principal_diag, "principal_diagnosis")

# High Potassium: Stacked Bar Chart
ggplot(high_KA_summary, aes(x = diagnosis_category, y = count, fill = age_group)) +
  geom_bar(stat = "identity", position = "dodge") +
  coord_flip() + # Flip for better readability if many categories exist
  labs(title = "Diagnosis Categories by Age Group (High Potassium)", 
       x = "Diagnosis Category", 
       y = "Count") +
  theme_minimal() +
  scale_fill_manual(values = c("<65" = "skyblue", "≥65" = "lightblue"))

# Low Potassium: Stacked Bar Chart
ggplot(low_KA_summary, aes(x = diagnosis_category, y = count, fill = age_group)) +
  geom_bar(stat = "identity", position = "dodge") +
  coord_flip() + # Flip for better readability if many categories exist
  labs(title = "Diagnosis Categories by Age Group (Low Potassium)", 
       x = "Diagnosis Category", 
       y = "Count") +
  theme_minimal() +
  scale_fill_manual(values = c("<65" = "pink", "≥65" = "lightcoral"))

# Remove variables to free up memory
rm(high_KA, low_KA, high_KA_summary_diag_category, low_KA_summary_diag_category,
   high_KA_summary_principal_diag, low_KA_summary_principal_diag)
gc()

# ------------------------------------------------------------------------------

##################################################
# Analysis of High and Low Leukocyte Count Cases #
##################################################

# Filter the dataset for rows where Leukn_first_value is greater than 400 (High Leukocyte Count)
high_leukn <- data[data$Leukn_first_value > 400, .SD, .SDcols = c(1:7, grep("Leukn", colnames(data)), ncol(data))]

# Filter the dataset for rows where Leukn_first_value is equal to 0 (Low Leukocyte Count)
low_leukn <- data[data$Leukn_first_value == 0, .SD, .SDcols = c(1:7, grep("Leukn", colnames(data)), ncol(data))]

# Sort the high leukocyte count data in descending order based on Leukn_first_value
high_leukn_sorted <- high_leukn[order(-high_leukn$Leukn_first_value), ]
# Sort the low leukocyte count data in ascending order based on Leukn_first_value
low_leukn_sorted <- low_leukn[order(low_leukn$Leukn_first_value), ]

# Print the relevant columns (3rd to 13th columns) of the sorted high leukocyte count cases
print(high_leukn_sorted[, 3:13])
# Print the relevant columns (3rd to 13th columns) of the sorted low leukocyte count cases
print(low_leukn_sorted[, 3:13])

# ------------------------------------------------------------------------------

##################################################
#   Analysis of Hemoglobin Levels (Hbn) Cases    #
##################################################

# Classify cases based on Hemoglobin levels
data[, anemia_category := fifelse(Hbn_first_value >= 100, "Mild",
                                  fifelse(Hbn_first_value >= 80 & Hbn_first_value < 100, "Moderate",
                                          fifelse(Hbn_first_value >= 65 & Hbn_first_value < 80, "Severe",
                                                  fifelse(Hbn_first_value < 65, "Life-threatening", NA_character_))))]

# Count cases in each anemia category
anemia_counts <- data[, .N, by = anemia_category]
anemia_counts[, percentage := round((N / sum(N)) * 100, 2)]

# Analyze principal diagnoses in each anemia category
anemia_diagnoses <- data[, .N, by = .(anemia_category, principal_diagnosis)]
anemia_diagnoses <- anemia_diagnoses[order(anemia_category, -N)]

# Print results
cat("\n### Anemia Category Counts and Percentages ###\n")
print(anemia_counts)

cat("\n### Principal Diagnoses by Anemia Category ###\n")
print(anemia_diagnoses)

# Glimpse: Show top 5 diagnoses for each anemia category
top_diagnoses_by_anemia <- anemia_diagnoses[order(anemia_category, -N), head(.SD, 5), by = anemia_category]

cat("\n### Top Diagnoses by Anemia Category ###\n")
print(top_diagnoses_by_anemia)

# Plot for the Anemia Category Distribution based on percentage
ggplot(anemia_counts, aes(x = anemia_category, y = percentage, fill = anemia_category)) +
  geom_bar(stat = "identity", show.legend = FALSE) +  # Create a bar plot
  scale_fill_manual(values = c("Mild" = "skyblue", 
                               "Moderate" = "skyblue", 
                               "Severe" = "skyblue", 
                               "Life-threatening" = "skyblue", 
                               "<NA>" = "gray")) +  # Color each anemia category differently
  labs(title = "Anemia Category Distribution", 
       x = "Anemia Category", 
       y = "Percentage (%)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels for better readability

# Analysis of Life-Threatening Anemia

# Filter Data for Life-Threatening Anemia
life_threatening_anemia <- data[anemia_category == "Life-threatening"]

# Plot the density
# Calculate the mode (peak) of the density plot
density_data <- density(life_threatening_anemia$Hbn_first_value, na.rm = TRUE)
peak_value <- density_data$x[which.max(density_data$y)]  # Find the x-value (Hbn_mean) at the peak
# Create a density plot with a dotted line at the peak and annotation using annotate()
ggplot(life_threatening_anemia, aes(x = Hbn_first_value)) +
  geom_density(fill = "skyblue", alpha = 0.5) +  # Create density plot
  geom_vline(aes(xintercept = peak_value), linetype = "dashed", color = "red") +  # Add dotted line at peak
  annotate("text", x = peak_value - 10, y = 0.080, label = paste("Hbn:", round(peak_value, 2)), 
           vjust = -1, color = "red") +  # Annotate the peak value
  labs(
    title = "Density Plot of Hbn_first_value for Life-Threatening Anemia",
    x = "Hemoglobin value (Hbn_first_value)",
    y = "Density"
  ) +
  theme_minimal()

# Filter rows where Hbn is 0
hbn_zero <- data[data$Hbn_first_value == 0, .SD, .SDcols = c(3:7, grep("Hbn", colnames(data)), ncol(data)-1)]
cat("\n### Rows with Hbn = 0 ###\n")
print(hbn_zero)

# Distribution of Diagnosis Categories
life_threatening_diag_category <- life_threatening_anemia[, .N, by = diagnosis_category]
setnames(life_threatening_diag_category, "N", "count")
setorder(life_threatening_diag_category, -count)

cat("\n### Diagnosis Categories for Life-Threatening Anemia ###\n")
print(life_threatening_diag_category)

# Distribution of Principal Diagnoses
life_threatening_principal_diag <- life_threatening_anemia[, .N, by = principal_diagnosis]
setnames(life_threatening_principal_diag, "N", "count")
setorder(life_threatening_principal_diag, -count)

cat("\n### Principal Diagnoses for Life-Threatening Anemia ###\n")
print(life_threatening_principal_diag)

# Top 10 Principal Diagnoses with Lowest Hbn Values (Excluding Hbn = 0)
lowest_hbn <- life_threatening_anemia[Hbn_first_value > 0, .(principal_diagnosis, Hbn_first_value)][order(Hbn_first_value)]

cat("\n### Top 10 Principal Diagnoses with Lowest Hbn Values (Excluding Hbn = 0) ###\n")
print(lowest_hbn)

# Calculate percentages for Diagnosis Categories in Life-Threatening Anemia
life_threatening_diag_category[, percentage := round((count / sum(count)) * 100, 2)]

# Plot for the Diagnosis Category Distribution based on percentage for life-threatening anemia cases
ggplot(life_threatening_diag_category, aes(x = percentage, y = diagnosis_category, fill = diagnosis_category)) +
  geom_bar(stat = "identity", show.legend = FALSE) +  # Create a bar plot
  scale_fill_manual(values = rep("skyblue", length(unique(life_threatening_diag_category$diagnosis_category)))) +  # Set color to skyblue for all categories
  labs(title = "Diagnosis Category Distribution for Life-Threatening Anemia", 
       x = "Percentage (%)", 
       y = "Diagnosis Category") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 0, hjust = 1))  # Rotate x-axis labels for better readability

# Remove variables that are no longer needed
rm(anemia_counts, anemia_diagnoses, top_diagnoses_by_anemia, life_threatening_diag_category, 
   life_threatening_principal_diag, lowest_hbn)
gc()

#----------------
# New Analysis: Cases with Hbn Max Value > 200

# Filter Data for Hbn Max Value > 200 - Cases where the maximum Hbn value exceeds 200
high_hbn_cases <- data[Hbn_max_value > 200]

# Distribution of Diagnosis Categories for Hbn Max Value > 200
high_hbn_diag_category <- high_hbn_cases[, .N, by = diagnosis_category]
setnames(high_hbn_diag_category, "N", "count")
setorder(high_hbn_diag_category, -count)

# Calculate percentages for Diagnosis Categories
high_hbn_diag_category[, percentage := round((count / sum(count)) * 100, 2)]

cat("\n### Diagnosis Categories for Hbn Max Value > 200 ###\n")
print(high_hbn_diag_category)

# Plot for the Diagnosis Category Distribution based on percentage
ggplot(high_hbn_diag_category, aes(x = percentage, y = diagnosis_category, fill = diagnosis_category)) +
  geom_bar(stat = "identity", show.legend = FALSE) +  # Create a bar plot
  scale_fill_manual(values = rep("skyblue", length(unique(high_hbn_diag_category$diagnosis_category)))) +  # Set color to skyblue for all categories
  labs(title = "Diagnosis Category Distribution for Hbn Max Value > 200", 
       x = "Percentage (%)", 
       y = "Diagnosis Category") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 0, hjust = 1))  # Rotate x-axis labels for better readability

# Distribution of Principal Diagnoses for Hbn Max Value > 200
high_hbn_principal_diag <- high_hbn_cases[, .N, by = principal_diagnosis]
setnames(high_hbn_principal_diag, "N", "count")
setorder(high_hbn_principal_diag, -count)

# Calculate percentages for Principal Diagnoses
high_hbn_principal_diag[, percentage := round((count / sum(count)) * 100, 2)]

cat("\n### Principal Diagnoses for Hbn Max Value > 200 ###\n")
print(high_hbn_principal_diag)

# Plot for the Principal Diagnosis Distribution based on percentage
ggplot(high_hbn_principal_diag, aes(x = reorder(principal_diagnosis, -percentage), y = percentage, fill = principal_diagnosis)) +
  geom_bar(stat = "identity", show.legend = FALSE) +  # Create a bar plot
  scale_fill_manual(values = rep("skyblue", length(unique(high_hbn_principal_diag$principal_diagnosis)))) +  # Set color to skyblue for all categories
  labs(title = "Principal Diagnoses Distribution for Hbn Max Value > 200", 
       x = "Principal Diagnosis", 
       y = "Percentage (%)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels for better readability

# Memory Cleanup: Remove temporary data objects to free up memory
rm(life_threatening_anemia, life_threatening_principal_diag, 
   lowest_hbn, high_hbn_cases, high_hbn_diag_category, high_hbn_principal_diag)
gc()

# ------------------------------------------------------------------------------

###################################################
# Age- and Sex-Based Trends in RBC Metrics (Eryn) #
###################################################

# Melt the dataset for easier faceted plotting
# Reshape the data to create long-format columns for RBC metrics (e.g., Eryn_mean, Eryn_median_value)
eryn_data <- melt(data, id.vars = c("age", "sex"), 
                  measure.vars = c("Eryn_mean", "Eryn_median_value", "Eryn_min_value", "Eryn_max_value", "Eryn_first_value"),
                  variable.name = "RBC_Metric", 
                  value.name = "Value")

# Create scatter plot for Eryn_mean vs. Age, colored by Sex
# Includes a dashed regression line to show trends in Eryn_mean with age
ggplot(data, aes(x = age, y = Eryn_mean, color = sex)) +
  geom_point(alpha = 0.6) + # Add data points with some transparency
  geom_smooth(method = "lm", se = FALSE, linetype = "dashed", color = "black") + # Add regression line
  labs(
    title = "Scatter Plot of Eryn_mean vs Age by Sex",
    x = "Age",
    y = "Eryn Mean",
    color = "Sex"
  ) +
  theme_minimal()

# Filter dataset to include only cases where Eryn_min_value equals 0
# This identifies cases with minimum RBC levels of zero
zero_Eryn <- data[data$Eryn_min_value == 0, .SD, .SDcols = c(3:7, grep("Eryn", colnames(data)), ncol(data))]

# Filter dataset for cases where Eryn_max_value is greater than 9
# This identifies cases with maximum RBC levels exceeding a high threshold
max_Eryn <- data[data$Eryn_max_value > 9, .SD, .SDcols = c(3:7, grep("Eryn", colnames(data)), ncol(data))]

# Print the filtered dataset for cases with Eryn_min_value == 0
print(zero_Eryn)
# Print the filtered dataset for cases with Eryn_max_value > 9
print(max_Eryn)

# ------------------------------------------------------------------------------

###################################################
#       Extreme values in platelets count         #
###################################################

# Filter dataset to include only cases where THZn_min_value equals 0
zero_THZn <- data[data$THZn_min_value == 0, .SD, .SDcols = c(3:7, grep("THZn", colnames(data)), ncol(data))]

# Filter dataset for cases where THZn_max_value is greater than 2000
max_THZn <- data[data$THZn_max_value > 2000, .SD, .SDcols = c(3:7, grep("THZn", colnames(data)), ncol(data))]

# Print the filtered dataset for cases with THZn_min_value == 0
print(zero_THZn)
# Print the filtered dataset for cases with THZn_max_value > 2000
print(max_THZn)

rm(zero_THZn, max_THZn)
# ------------------------------------------------------------------------------

###################################################
#                  Outlier for NAT                #
###################################################
# Filter dataset for cases where NAT_max_value is greater than 191
max_NAT <- data[data$NAT_max_value > 191, .SD, .SDcols = c(3:7, grep("NAT", colnames(data)), ncol(data))]
print(max_NAT)

rm(max_NAT)

# ------------------------------------------------------------------------------

###################################################
#                 Outliers for INRiH               #
###################################################

# Filter dataset to include only cases where INRiH_min_value smaller than 0.8
low_INRiH <- data[data$INRiH_min_value < 0.8, .SD, .SDcols = c(3:7, grep("INRiH", colnames(data)), ncol(data))]

# Filter dataset for cases where INRiH_max_value is greater than 10
high_INRiH <- data[data$INRiH_max_value > 10, .SD, .SDcols = c(3:7, grep("INRiH", colnames(data)), ncol(data))]

# Print the filtered dataset for cases with INRiH_min_value smaller than 0.8
print(low_INRiH)
# Print the filtered dataset for cases with INRiH_max_value is greater than 10
print(high_INRiH)

rm(low_INRiH, high_INRiH)

# ------------------------------------------------------------------------------

###################################################
#               Extremely high QUHD               #
###################################################
# Filter dataset for cases where QUHD_max_value is greater than 120
high_QUHD <- data[data$QUHD_max_value > 120, .SD, .SDcols = c(3:7, grep("QUHD", colnames(data)), ncol(data))]
# Order by QUHD_max_value in descending order
setorder(high_QUHD, -QUHD_max_value)
# Print the filtered dataset
print(high_QUHD[high_QUHD$QUHD_max_value > 170])
rm(high_QUHD)

# ------------------------------------------------------------------------------

###################################################
#    Identifying Extremes in Glucose Levels       #
###################################################
# Filter dataset for cases where GL_min_value is lower than 2.8 (a typical threshold for hypoglycemia)
low_GL <- data[data$GL_min_value < 2.8, .SD, .SDcols = c(3:7, grep("GL", colnames(data)), ncol(data))]
# Order by GL_min_value in ascending order (to see the lowest values)
setorder(low_GL, GL_min_value)
# Print the first few rows of the filtered dataset
print(head(low_GL))

# Filter dataset for cases where GL_max_value is greater than 50 (a very high glucose level)
high_GL <- data[data$GL_max_value > 50, .SD, .SDcols = c(3:7, grep("GL", colnames(data)), ncol(data))]
# Order by GL_max_value in descending order (to see the highest values)
setorder(high_GL, -GL_max_value)
# Print the first few rows of the filtered dataset
print(head(high_GL))

rm(low_GL, high_GL)

# ------------------------------------------------------------------------------

###################################################
#   Identifying Extremes in creatinine values     #
###################################################
# Filter dataset for cases where CR_min_value is lower than a low threshold (e.g., < 10 µmol/L, indicating very low creatinine)
low_CR <- data[data$CR_min_value < 10, .SD, .SDcols = c(3:7, grep("CR_", colnames(data)), ncol(data)-1)]
# Order by CR_min_value in ascending order (to see the lowest values)
setorder(low_CR, CR_min_value)
# Print the number of rows in the filtered dataset
print(nrow(low_CR))
# Print the first few rows of the filtered dataset
print(head(low_CR))

# Filter dataset for cases where CR_max_value is greater than a high threshold (e.g., > 2000 µmol/L, indicating extremely high creatinine levels)
high_CR <- data[data$CR_max_value > 2000, .SD, .SDcols = c(3:7, grep("CR_", colnames(data)), ncol(data)-1)]
# Order by CR_max_value in descending order (to see the highest values)
setorder(high_CR, -CR_max_value)
# Print the number of rows in the filtered dataset
print(nrow(high_CR))
# Print the first few rows of the filtered dataset
print((high_CR))

rm(low_CR, high_CR)

# ------------------------------------------------------------------------------

###################################################
#               Extremely high NRBCmn             #
###################################################
# Filter dataset for cases where NRBCmn_max_value is greater than a high threshold (100/100 leuk)
high_NRBCmn <- data[data$NRBCmn_max_value > 100, .SD, .SDcols = c(3:7, grep("NRBCmn", colnames(data)), ncol(data))]
# Order by NRBCmn_max_value in descending order (to see the highest values)
setorder(high_NRBCmn, -NRBCmn_max_value)
# Print the first few rows of the filtered dataset
print((high_NRBCmn))

rm(high_NRBCmn)

# ------------------------------------------------------------------------------

###################################################
#    Identifying Extremes in eGFR Values         #
###################################################
# Filter dataset for cases where eGFR (EPIGFR) is lower than 15 (severe kidney dysfunction threshold)
low_eGFR <- data[data$EPIGFR_min_value < 15, .SD, .SDcols = c(3:7, grep("EPIGFR_", colnames(data)), ncol(data))]
# Order by EPIGFR in ascending order (to see the lowest eGFR values)
setorder(low_eGFR, EPIGFR_min_value)
# Print the first few rows of the filtered dataset
print(head(low_eGFR))

# Filter dataset for cases where eGFR (EPIGFR) is greater than 200 (extremely high eGFR)
high_eGFR <- data[data$EPIGFR_max_value > 200, .SD, .SDcols = c(3:7, grep("EPIGFR_", colnames(data)), ncol(data))]
# Order by EPIGFR in descending order (to see the highest eGFR values)
setorder(high_eGFR, -EPIGFR_max_value)
# Print the first few rows of the filtered dataset
print(head(high_eGFR))

rm(low_eGFR, high_eGFR)

# How many entries with max_value > 140:
nrow(data[data$EPIGFR_max_value > 140, .SD, .SDcols = c(3:7, grep("EPIGFR_", colnames(data)), ncol(data))])

# ------------------------------------------------------------------------------

###################################################
#       Extremely high entries for Quicks         #
###################################################
# Filter dataset for cases where Quicks is greater than 400 (extremely high Quicks values)
high_Quicks <- data[data$Quicks_max_value > 400, .SD, .SDcols = c(3:7, grep("Quicks", colnames(data)), ncol(data))]

# Order by Quicks in descending order (to see the highest Quicks values)
setorder(high_Quicks, -Quicks_max_value)

# Print the number of rows in the filtered dataset
print(nrow(high_Quicks))
# Print the first few rows of the filtered dataset
print(high_Quicks)

# Clean up temporary variables
rm(high_Quicks)

# ------------------------------------------------------------------------------

###################################################
#     Identifying Extremes in ASAT Values         #
###################################################
# Filter dataset for cases where ASAT_max_value is greater than 5000 U/L
high_ASAT_max <- data[data$ASAT_max_value > 5000, .SD, .SDcols = c(3:7, grep("ASAT", colnames(data)), ncol(data))]
# Order by ASAT_max_value in descending order (to see the highest ASAT values)
setorder(high_ASAT_max, -ASAT_max_value)
# Print the number of rows in the filtered dataset
print(nrow(high_ASAT_max))
# Print the first few rows of the filtered dataset
print(head(high_ASAT_max))

# Filter dataset for cases where ASAT_min_value is equal to 0 (possible errors or missing values)
zero_ASAT_min <- data[data$ASAT_min_value == 0, .SD, .SDcols = c(3:7, grep("ASAT", colnames(data)), ncol(data))]
# Order by ASAT_min_value in ascending order (to see the zeros)
setorder(zero_ASAT_min, ASAT_min_value)
# Print the number of rows in the filtered dataset
print(nrow(zero_ASAT_min))
# Print the first few rows of the filtered dataset
print(head(zero_ASAT_min))

# Filter dataset for cases where ASAT_min_value is smaller than 8 
low_ASAT_min <- data[data$ASAT_min_value < 8, .SD, .SDcols = c(3:7, grep("ASAT", colnames(data)), ncol(data))]
# Order by ASAT_min_value in ascending order 
setorder(low_ASAT_min, ASAT_min_value)
# Print the number of rows in the filtered dataset
print(nrow(low_ASAT_min))
# Print the first few rows of the filtered dataset
print(head(low_ASAT_min))

# Clean up variables to free memory
rm(high_ASAT_max, zero_ASAT_min, low_ASAT_min)

# ------------------------------------------------------------------------------

###################################################
#     Identifying Extremes in ALAT Values         #
###################################################
# Filter dataset for cases where ALAT_max_value is greater than 10000 U/L
high_ALAT <- data[data$ALAT_max_value > 10000, .SD, .SDcols = c(3:7, grep("ALAT_", colnames(data)), ncol(data))]
# Order by ALAT_max_value in descending order (to see the highest ALAT values)
setorder(high_ALAT, -ALAT_max_value)
# Print the number of rows in the filtered dataset
print(nrow(high_ALAT))
# Print the first few rows of the filtered dataset
print(high_ALAT)

# Filter dataset for cases where ALAT_min_value is 0
low_ALAT <- data[data$ALAT_min_value == 0, .SD, .SDcols = c(3:7, grep("ALAT_", colnames(data)), ncol(data))]
# Order by ALAT_min_value in ascending order (to see cases with zero values)
setorder(low_ALAT, ALAT_min_value)
# Print the number of rows in the filtered dataset
print(nrow(low_ALAT))
# Print the first few rows of the filtered dataset
print(head(low_ALAT))

# Clean up
rm(high_ALAT, low_ALAT)

# ------------------------------------------------------------------------------

###################################################
#         Extremely high entries for GGT          #
###################################################
# Filter dataset for cases where GGT_max_value is greater than 2000 (extremely high GGT values)
high_GGT <- data[data$GGT_max_value > 2000, .SD, .SDcols = c(3:7, grep("GGT_", colnames(data)), ncol(data))]
# Order by GGT_max_value in descending order (to see the highest GGT values)
setorder(high_GGT, -GGT_max_value)
# Print the number of rows in the filtered dataset
print(nrow(high_GGT))
# Print the first few rows of the filtered dataset
print(head(high_GGT))

# Filter dataset for cases where GGT_min_value is lower than 5 (extremely low GGT values)
low_GGT <- data[data$GGT_min_value < 5, .SD, .SDcols = c(3:7, grep("GGT_", colnames(data)), ncol(data)-1)]
# Order by GGT_max_value in descending order (to see the highest GGT values)
setorder(low_GGT, GGT_min_value)
# Print the number of rows in the filtered dataset
print(nrow(low_GGT))
# Print the first few rows of the filtered dataset
print(head(low_GGT))

rm(high_GGT, low_GGT)

# ------------------------------------------------------------------------------

###################################################
#     Identifying Extremes in UREA Values         #
###################################################
# Filter out rows where UREA_max_value exceeds 50 (chosen threshold for extreme values)
high_UREA <- data[data$UREA_max_value > 50, .SD, .SDcols = c(3:7, grep("UREA_", colnames(data)), ncol(data))]
# Print the number of rows in the filtered dataset
print(nrow(high_UREA))
# Print the first few rows of the high UREA dataset
print(head(high_UREA))


# Filter out rows where UREA_min_value equals 0 (for checking low UREA values)
low_UREA <- data[data$UREA_min_value == 0, .SD, .SDcols = c(3:7, grep("UREA_", colnames(data)), ncol(data))]
# Print the number of rows in the filtered dataset
print(nrow(low_UREA))
# Print the first few rows of the low UREA dataset
print(head(low_UREA))

rm(high_UREA, low_UREA)

# ------------------------------------------------------------------------------

###################################################
#     Identifying Extremes in CK Values           #
###################################################
# Filter out rows where CK_max_value exceeds 200,000 (chosen threshold for extreme high values)
high_CK <- data[data$CK_max_value > 200000, .SD, .SDcols = c(3:7, grep("CK_", colnames(data)), ncol(data))]
# Print the number of rows in the filtered dataset for high CK values
print(nrow(high_CK))
# Print the first few rows of the high CK dataset
print(head(high_CK))

# Filter out rows where CK_min_value is less than 5 (for checking low CK values)
low_CK <- data[data$CK_min_value < 5, .SD, .SDcols = c(3:7, grep("CK_", colnames(data)), ncol(data))]
# Print the number of rows in the filtered dataset for low CK values
print(nrow(low_CK))
# Print the first few rows of the low CK dataset
print(head(low_CK))

rm(high_CK, low_CK)  # Clean up the temporary datasets

# ------------------------------------------------------------------------------

###################################################
#               Extremely low CA                 #
###################################################
# Filter dataset for cases where CA_min_value is below a low threshold (e.g., 1.0 mmol/L)
low_CA <- data[data$CA_min_value < 1.0, .SD, .SDcols = c(3:7, grep("CA_", colnames(data)), ncol(data))]
# Order by CA_min_value in ascending order (to see the lowest values)
setorder(low_CA, CA_min_value)
# Print the number of rows in the filtered dataset
print(nrow(low_CA))
# Print the first few rows of the filtered dataset
print(low_CA)
# Clean up the filtered dataset object after use
rm(low_CA)

# ------------------------------------------------------------------------------

###################################################
#         Extremely high entries for EC3.U        #
###################################################
# Filter dataset for cases where EC3.U_max_value is greater than 10 (extremely high EC3.U values)
high_EC3U <- data[data$EC3.U_max_value > 10, .SD, .SDcols = c(3:7, grep("EC3.U", colnames(data)), ncol(data))]
# Order by EC3.U_max_value in descending order (to see the highest EC3.U values)
setorder(high_EC3U, -EC3.U_max_value)
# Print the number of rows in the filtered dataset
print(nrow(high_EC3U))
# Print the first few rows of the filtered dataset
print(head(high_EC3U))

rm(high_EC3U)

# ------------------------------------------------------------------------------

######################################################################
#        Extremely high entries for GLUC3 (Glucose in urine)         #
######################################################################
# Filter dataset for cases where GLUC3_max_value exceeds 30 (extremely high glucose values)
high_GLUC3 <- data[data$GLUC3_max_value > 30, .SD, .SDcols = c(3:7, grep("GLUC3_", colnames(data)), ncol(data))]
# Order by GLUC3_max_value in descending order (to see the highest values)
setorder(high_GLUC3, -GLUC3_max_value)
# Print the number of rows in the filtered dataset
print(nrow(high_GLUC3))
# Print the first few rows of the filtered dataset
print(head(high_GLUC3))

rm(high_GLUC3)

# ------------------------------------------------------------------------------

#########################################################################
#         Extremely high entries for BI3.U (bilirubin in urine)         #
#########################################################################
# Filter dataset for cases where BI3.U_max_value is greater than 20 (extremely high BI3.U values)
high_BI3U <- data[data$BI3.U_max_value > 20, .SD, .SDcols = c(3:7, grep("BI3.U", colnames(data)), ncol(data))]
# Order by BI3.U_max_value in descending order (to see the highest BI3.U values)
setorder(high_BI3U, -BI3.U_max_value)
# Print the number of rows in the filtered dataset
print(nrow(high_BI3U))
# Print the first few rows of the filtered dataset
print(head(high_BI3U))

rm(high_BI3U)

# ------------------------------------------------------------------------------

####################################################
#         Extremely high entries for URO3          #
####################################################
# Filter dataset for cases where URO3_max_value is greater than a threshold (e.g., 100)
high_URO3 <- data[data$URO3_max_value > 100, .SD, .SDcols = c(3:7, grep("URO3_", colnames(data)), ncol(data))]
# Order by URO3_max_value in descending order to see the highest URO3 values
setorder(high_URO3, -URO3_max_value)
# Print the number of rows in the filtered dataset
print(nrow(high_URO3))
# Print the first few rows of the filtered dataset
print(head(high_URO3))
# Remove the variable after use to free up memory
rm(high_URO3)

# ------------------------------------------------------------------------------

####################################################
#         Extremely high entries for LK3.U         #
####################################################
# Filter dataset for cases where LK3.U_max_value is greater than a certain threshold (e.g., 50)
high_LK3 <- data[data$LK3.U_max_value > 50, .SD, .SDcols = c(3:7, grep("LK3.U", colnames(data)), ncol(data))]
# Order by LK3.U_max_value in descending order to view the highest values first
setorder(high_LK3, -LK3.U_max_value)
# Print the number of rows in the filtered dataset
print(nrow(high_LK3))
# Print the first few rows of the filtered dataset
print(head(high_LK3))
# Optionally remove the high_LK3 data frame to free up memory
rm(high_LK3)

# ------------------------------------------------------------------------------

####################################################
#         Extremely high entries for NITR3         #
####################################################
# Filter the dataset for high nitrite values (e.g., greater than 1 or a threshold of your choice)
high_nitrites <- data[data$NITR3_max_value > 0, .SD, .SDcols = c(3:7, grep("NITR3", colnames(data)), ncol(data))]
# Order by NITR3 value in descending order to see the highest nitrite values
setorder(high_nitrites, -NITR3_max_value)
# Print the number of rows with high nitrite values
print(nrow(high_nitrites))
# Print the first few rows of the filtered dataset to check the details
print(head(high_nitrites))
# Optionally remove the temporary high_nitrites dataset to free memory
rm(high_nitrites)

# ------------------------------------------------------------------------------

####################################################
#         Extremely high entries for KETO3         #
####################################################
# Filter the dataset for high KETO3 values (e.g., greater than 10)
high_keto3 <- data[data$KETO3_max_value > 10, .SD, .SDcols = c(3:7, grep("KETO3", colnames(data)), ncol(data))]
# Order by KETO3_max_value in descending order to see the highest KETO3 values
setorder(high_keto3, -KETO3_max_value)
# Print the number of rows with high KETO3 values
print(nrow(high_keto3))
# Print the first few rows of the filtered dataset to check the details
print(head(high_keto3))
# Optionally remove the temporary high_keto3 dataset to free memory
rm(high_keto3)

# ------------------------------------------------------------------------------

####################################################
#         Extremely high entries for SPEZ3         #
####################################################
# Filter the dataset for high SPEZ3 values 
high_spez3 <- data[data$SPEZ3_max_value > 1.030, .SD, .SDcols = c(3:7, grep("SPEZ3", colnames(data)), ncol(data))]
# Order by SPEZ3_max_value in descending order to see the highest specific gravity values
setorder(high_spez3, -SPEZ3_max_value)
# Print the number of rows with high SPEZ3 values
print(nrow(high_spez3))
# Print the first few rows of the filtered dataset to check the details
print(head(high_spez3))
# Optionally remove the temporary high_spez3 dataset to free memory
rm(high_spez3)

# ------------------------------------------------------------------------------

############################################################
#         Extremely high entries for BIg (Bilirubin)       #
############################################################
# Define the threshold for high BIg values (e.g., greater than a plausible high limit such as 50 or 100 µmol/L)
threshold <- 50
# Filter the dataset for high BIg values
high_big <- data[data$BIg_max_value > threshold, .SD, .SDcols = c(3:7, grep("BIg", colnames(data)), ncol(data))]
# Order by BIg_max_value in descending order to see the highest BIg values
setorder(high_big, -BIg_max_value)
# Print the number of rows with high BIg values
print(nrow(high_big))
# Print the first few rows of the filtered dataset to check the details
print(head(high_big))

# Optionally remove the temporary high_big dataset to free memory
rm(high_big)

# ------------------------------------------------------------------------------

####################################################################
#     Identifying Extremes in AP (Alkaline Phosphatase) Values     #
####################################################################
# Filter out rows where AP_max_value exceeds a chosen threshold (e.g., 1000 U/L for high values)
high_AP <- data[data$AP_max_value > 1000, .SD, .SDcols = c(3:7, grep("AP_", colnames(data)), ncol(data))]
# Order by AP_max_value in descending order to see the highest AP values
setorder(high_AP, -AP_max_value)
# Print the number of rows in the filtered dataset for high AP values
print(nrow(high_AP))
# Print the first few rows of the high AP dataset to check the details
print(head(high_AP))

# Filter out rows where AP_min_value is less than a threshold (e.g., 10 U/L for low values)
low_AP <- data[data$AP_min_value < 10, .SD, .SDcols = c(3:7, grep("AP_", colnames(data)), ncol(data))]
# Order by AP_min_value in asccending order to see the lowest AP values
setorder(low_AP, AP_min_value)
# Print the number of rows in the filtered dataset for low PA values
print(nrow(low_AP))
# Print the first few rows of the low AP dataset to check the details
print(head(low_AP))

# Clean up the temporary high and low AP datasets to free memory
rm(high_AP, low_AP)

# ------------------------------------------------------------------------------

####################################################################
#     Identifying Extremes in Neutrophile counts (NEUm#n)          #
####################################################################
# Filter out rows where NEUm.n_max_value exceeds a chosen threshold 
high_NEUm.n <- data[data$NEUm.n_max_value > 50, .SD, .SDcols = c(3:7, grep("NEUm.n_", colnames(data)), ncol(data)-1)]
# Order by NEUm.n_max_value in descending order to see the highest NEUm.n values
setorder(high_NEUm.n, -NEUm.n_max_value)
# Print the number of rows in the filtered dataset for high NEUm.n values
print(nrow(high_NEUm.n))
# Print the first few rows of the high NEUm.n dataset to check the details
print(high_NEUm.n)

# Filter out rows where NEUm.n_min_value is less than a threshold 
low_NEUm.n <- data[data$NEUm.n_min_value < 1, .SD, .SDcols = c(3:7, grep("NEUm.n_", colnames(data)), ncol(data)-1)]
# Order by NEUm.n_min_value in asccending order to see the lowest NEUm.n values
setorder(low_NEUm.n, NEUm.n_min_value)
# Print the number of rows in the filtered dataset for low NEUm.n values
print(nrow(low_NEUm.n))
# Print the first few rows of the low NEUm.n dataset to check the details
print(head(low_NEUm.n))

# Clean up the temporary high and low AP datasets to free memory
rm(high_NEUm.n, low_NEUm.n)

# ------------------------------------------------------------------------------

####################################################################
#     Identifying Extremes in Lymphocyte counts (LYMm#n)           #
####################################################################
# Filter out rows where LYMm.n_max_value exceeds a chosen threshold 
high_LYMm.n <- data[data$LYMm.n_max_value > 50, .SD, .SDcols = c(3:7, grep("LYMm.n_", colnames(data)), ncol(data)-1)]
# Order by LYMm.n_max_value in descending order to see the highest LYMm.n values
setorder(high_LYMm.n, -LYMm.n_max_value)
# Print the number of rows in the filtered dataset for high LYMm.n values
print(nrow(high_LYMm.n))
# Print the first few rows of the high LYMm.n dataset to check the details
print(high_LYMm.n)

# Filter out rows where LYMm.n_min_value is less than a threshold 
low_LYMm.n <- data[data$LYMm.n_min_value < 1, .SD, .SDcols = c(3:7, grep("LYMm.n_", colnames(data)), ncol(data)-1)]
# Order by LYMm.n_min_value in asccending order to see the lowest LYMm.n values
setorder(low_LYMm.n, LYMm.n_min_value)
# Print the number of rows in the filtered dataset for low LYMm.n values
print(nrow(low_LYMm.n))
# Print the first few rows of the low LYMm.n dataset to check the details
print(head(low_LYMm.n))

# Clean up the temporary high and low LYMm.n datasets to free memory
rm(high_LYMm.n, low_LYMm.n)

# ------------------------------------------------------------------------------

####################################################################
#     Identifying Extremes in Eosinophil counts (EOSm#n)           #
####################################################################
# Filter out rows where EOSm.n_max_value exceeds a chosen threshold 
high_EOSm.n <- data[data$EOSm.n_max_value > 7, .SD, .SDcols = c(3:7, grep("EOSm.n_", colnames(data)), ncol(data)-1)]
# Order by EOSm.n_max_value in descending order to see the highest EOSm.n values
setorder(high_EOSm.n, -EOSm.n_max_value)
# Print the number of rows in the filtered dataset for high EOSm.n values
print(nrow(high_EOSm.n))
# Print the first few rows of the high EOSm.n dataset to check the details
print(high_EOSm.n)

# Filter out rows where EOSm.n_min_value is less than a threshold 
low_EOSm.n <- data[data$EOSm.n_min_value < 1, .SD, .SDcols = c(3:7, grep("EOSm.n_", colnames(data)), ncol(data)-1)]
# Order by EOSm.n_min_value in asccending order to see the lowest LYMm.n values
setorder(low_EOSm.n, EOSm.n_min_value)
# Print the number of rows in the filtered dataset for low EOSm.n values
print(nrow(low_EOSm.n))
# Print the first few rows of the low EOSm.n dataset to check the details
print(head(low_EOSm.n))

# Clean up the temporary high and low EOSm.n datasets to free memory
rm(high_EOSm.n, low_EOSm.n)

# ------------------------------------------------------------------------------

####################################################################
#     Identifying Extremes in Monocyte counts (MONm#n)             #
####################################################################
# Filter out rows where MONm.n_max_value exceeds a chosen threshold 
high_MONm.n <- data[data$MONm.n_max_value > 40, .SD, .SDcols = c(3:7, grep("MONm.n_", colnames(data)), ncol(data)-1)]
# Order by MONm.n_max_value in descending order to see the highest MONm.n values
setorder(high_MONm.n, -MONm.n_max_value)
# Print the number of rows in the filtered dataset for high MONm.n values
print(nrow(high_MONm.n))
# Print the first few rows of the high MONm.n dataset to check the details
print(high_MONm.n)

# Filter out rows where EOSm.n_min_value is less than a threshold 
low_MONm.n <- data[data$EOSm.n_min_value < 0.2, .SD, .SDcols = c(3:7, grep("MONm.n_", colnames(data)), ncol(data)-1)]
# Order by MONm.n_min_value in asccending order to see the lowest MONm.n values
setorder(low_MONm.n, MONm.n_min_value)
# Print the number of rows in the filtered dataset for low MONm.n values
print(nrow(low_MONm.n))
# Print the first few rows of the low MONm.n dataset to check the details
print(head(low_MONm.n))

# Clean up the temporary high and low MONm.n datasets to free memory
rm(high_MONm.n, low_MONm.n)

# ------------------------------------------------------------------------------

####################################################################
#     Identifying Extremes in Basophils count (BASm#n)             #
####################################################################
# Filter out rows where BASm.n_max_value exceeds a chosen threshold 
high_BASm.n <- data[data$BASm.n_max_value > 1, .SD, .SDcols = c(3:7, grep("BASm.n_", colnames(data)), ncol(data)-1)]
# Order by BASm.n_max_value in descending order to see the highest BASm.n values
setorder(high_BASm.n, -BASm.n_max_value)
# Print the number of rows in the filtered dataset for high BASm.n values
print(nrow(high_BASm.n))
# Print the first few rows of the high BASm.n dataset to check the details
print(high_BASm.n)

# Filter out rows where BASm.n_min_value is less than a threshold 
low_BASm.n <- data[data$BASm.n_min_value < 0.02, .SD, .SDcols = c(3:7, grep("BASm.n_", colnames(data)), ncol(data)-1)]
# Order by BASm.n_min_value in asccending order to see the lowest BASm.n values
setorder(low_BASm.n, BASm.n_min_value)
# Print the number of rows in the filtered dataset for low BASm.n values
print(nrow(low_BASm.n))
# Print the first few rows of the low BASm.n dataset to check the details
print(head(low_BASm.n))

# Clean up the temporary high and low MONm.n datasets to free memory
rm(high_BASm.n, low_BASm.n)

# ------------------------------------------------------------------------------

####################################################################
#     Identifying Extremes in Immature Granulocytes (IGm#n)        #
####################################################################
# Filter out rows where IGm.n_max_value exceeds a chosen threshold 
high_IGm.n <- data[data$IGm.n_max_value > 10, .SD, .SDcols = c(3:7, grep("IGm.n_", colnames(data)), ncol(data)-1)]
# Order by IGm.n_max_value in descending order to see the highest IGm.n values
setorder(high_IGm.n, -IGm.n_max_value)
# Print the number of rows in the filtered dataset for high IGm.n values
print(nrow(high_IGm.n))
# Print the first few rows of the high IGm.n dataset to check the details
print(high_IGm.n)

# Filter out rows where IGm.n_min_value is less than a threshold 
low_IGm.n <- data[data$IGm.n_min_value == 0, .SD, .SDcols = c(3:7, grep("IGm.n_", colnames(data)), ncol(data)-1)]
# Order by IGm.n_min_value in asccending order to see the lowest IGm.n values
setorder(low_IGm.n, IGm.n_min_value)
# Print the number of rows in the filtered dataset for low IGm.n values
print(nrow(low_IGm.n))
# Print the first few rows of the low IGm.n dataset to check the details
print(head(low_IGm.n))

# Clean up the temporary high and low IGm.n datasets to free memory
rm(high_IGm.n, low_IGm.n)

# ------------------------------------------------------------------------------

########################################################################
#     Identifying Extremes in High-sensitivity troponin T (TNThsn)     #
########################################################################
# Filter out rows where TNThsn_max_value exceeds a chosen threshold 
high_TNThsn <- data[data$TNThsn_max_value > 50000, .SD, .SDcols = c(3:7, grep("TNThsn_", colnames(data)), ncol(data)-1)]
# Order by TNThsn_max_value in descending order to see the highest TNThsn values
setorder(high_TNThsn, -TNThsn_max_value)
# Print the number of rows in the filtered dataset for high TNThsn values
print(nrow(high_TNThsn))
# Print the first few rows of the high TNThsn dataset to check the details
print(high_TNThsn)

# Filter out rows where TNThsn_min_value is less than a threshold 
low_TNThsn <- data[data$TNThsn_min_value == 0, .SD, .SDcols = c(3:7, grep("TNThsn_", colnames(data)), ncol(data)-1)]
# Order by TNThsn_min_value in asccending order to see the lowest TNThsn values
setorder(low_TNThsn, TNThsn_min_value)
# Print the number of rows in the filtered dataset for low IGm.n values
print(nrow(low_TNThsn))
# Print the first few rows of the low IGm.n dataset to check the details
print(head(low_TNThsn))

# Clean up the temporary high and low IGm.n datasets to free memory
rm(high_TNThsn, low_TNThsn)

# ------------------------------------------------------------------------------

############################################################
#        Extremely high entries for H-Se, I-Se, L-Se       #
############################################################
# Filter out rows where H.Se_max_value exceeds a chosen threshold 
high_H.Se <- data[data$H.Se_max_value > 1000, .SD, .SDcols = c(3:7, grep("H.Se_", colnames(data)), ncol(data)-1)]
# Order by H.Se_max_value in descending order to see the highest H.Se values
setorder(high_H.Se, -H.Se_max_value)
# Print the number of rows in the filtered dataset for high H.Se values
print(nrow(high_H.Se))
# Print the first few rows of the high H.Se dataset to check the details
print(high_H.Se)

# Filter out rows where I.Se_max_value exceeds a chosen threshold 
high_I.Se <- data[data$I.Se_max_value > 200, .SD, .SDcols = c(3:7, grep("I.Se_", colnames(data)), ncol(data)-1)]
# Order by I.Se_max_value in descending order to see the highest I.Se values
setorder(high_I.Se, -I.Se_max_value)
# Print the number of rows in the filtered dataset for high I.Se values
print(nrow(high_I.Se))
# Print the first few rows of the high I.Se dataset to check the details
print(high_I.Se)

# Filter out rows where L.Se_max_value exceeds a chosen threshold 
high_L.Se <- data[data$L.Se_max_value > 500, .SD, .SDcols = c(3:7, grep("L.Se_", colnames(data)), ncol(data)-1)]
# Order by L.Se_max_value in descending order to see the highest L.Se values
setorder(high_L.Se, -L.Se_max_value)
# Print the number of rows in the filtered dataset for high L.Se values
print(nrow(high_L.Se))
# Print the first few rows of the high L.Se dataset to check the details
print(high_L.Se)

rm(high_H.Se, high_I.Se, high_L.Se)
# ------------------------------------------------------------------------------

####################################################################
#             Identifying Extremes in Lactate (LACT)               #   
####################################################################
# Filter out rows where LACT_max_value exceeds a chosen threshold 
high_LACT <- data[data$LACT_max_value > 15, .SD, .SDcols = c(3:7, grep("LACT_", colnames(data)), ncol(data)-1)]
# Order by LACT_max_value in descending order to see the highest LACT values
setorder(high_LACT, -LACT_max_value)
# Print the number of rows in the filtered dataset for high LACT values
print(nrow(high_LACT))
# Print the first few rows of the high LACT dataset to check the details
print(head(high_LACT))

# Filter out rows where LACT_min_value is less than a threshold 
low_LACT <- data[data$LACT_min_value < 0.5, .SD, .SDcols = c(3:7, grep("LACT_", colnames(data)), ncol(data)-1)]
# Order by LACT_min_value in asccending order to see the lowest LACT values
setorder(low_LACT, LACT_min_value)
# Print the number of rows in the filtered dataset for low LACT values
print(nrow(low_LACT))
# Print the first few rows of the low LACT dataset to check the details
print(head(low_LACT))

# Clean up the temporary high and low LYMm datasets to free memory
rm(high_LACT, low_LACT)

# ------------------------------------------------------------------------------

####################################################################
#        Identifying Extremes in Total hemoglobin (tHb)            #   
####################################################################
# Filter out rows where tHb_min_value is less than a threshold 
low_tHb <- data[data$tHb_min_value < 30, .SD, .SDcols = c(3:7, grep("tHb_", colnames(data)), ncol(data)-1)]
# Order by tHb_min_value in ascending order to see the lowest tHb values
setorder(low_tHb, tHb_min_value)
# Print the number of rows in the filtered dataset for low tHb values
print(nrow(low_tHb))
# Print the first few rows of the low tHb dataset to check the details
print(head(low_tHb))

# Clean up the temporary high and low tHb datasets to free memory
rm(low_tHb)

# ------------------------------------------------------------------------------

####################################################################
#        Identifying Extremes in Carboxyhemoglobin (CO.HB)         #
####################################################################
# Filter out rows where CO.HB_max_value exceeds a chosen threshold 
high_CO.HB <- data[data$CO.HB_max_value > 20, .SD, .SDcols = c(3:7, grep("CO.HB_", colnames(data)), ncol(data)-1)]
# Order by CO.HB_max_value in descending order to see the highest CO.HB values
setorder(high_CO.HB, -CO.HB_max_value)
# Print the number of rows in the filtered dataset for high CO.HB values
print(nrow(high_CO.HB))
# Print the first few rows of the high CO.HB dataset to check the details
print(head(high_CO.HB))

# Clean up the temporary high and low tHb datasets to free memory
rm(high_CO.HB)

# ------------------------------------------------------------------------------

####################################################################
#           Identifying Extremes in Methemoglobin (MTHB)           #   
####################################################################
# Filter out rows where MTHB_max_value exceeds a chosen threshold 
high_MTHB <- data[data$MTHB_max_value > 4, .SD, .SDcols = c(3:7, grep("MTHB_", colnames(data)), ncol(data)-1)]
# Order by MTHB_max_value in descending order to see the highest MTHB values
setorder(high_MTHB, -MTHB_max_value)
# Print the number of rows in the filtered dataset for high MTHB values
print(nrow(high_MTHB))
# Print the first few rows of the high MTHB dataset to check the details
print(high_MTHB)

# Clean up the temporary high and low tHb datasets to free memory
rm(high_MTHB)

# ------------------------------------------------------------------------------

####################################################################
#        Identifying Extremes in Body temperature (Tbga)           #  
####################################################################
# Filter out rows where Tbga_max_value exceeds a chosen threshold 
high_Tbga <- data[data$Tbga_max_value > 42, .SD, .SDcols = c(3:7, grep("Tbga_", colnames(data)), ncol(data)-1)]
# Order by Tbga_max_value in descending order to see the highest Tbga values
setorder(high_Tbga, -Tbga_max_value)
# Print the number of rows in the filtered dataset for high Tbga values
print(nrow(high_Tbga))
# Print the first few rows of the high Tbga dataset to check the details
print(head(high_Tbga))

# Filter out rows where Tbga_min_value is less than a threshold 
low_Tbga <- data[data$Tbga_min_value < 28, .SD, .SDcols = c(3:7, grep("Tbga_", colnames(data)), ncol(data)-1)]
# Order by Tbga_min_value in ascending order to see the lowest Tbga values
setorder(low_Tbga, Tbga_min_value)
# Print the number of rows in the filtered dataset for low Tbga values
print(nrow(low_Tbga))
# Print the first few rows of the low Tbga dataset to check the details
print(head(low_Tbga))

# Clean up the temporary high and low Tbga datasets to free memory
rm(high_Tbga, low_Tbga)

# ------------------------------------------------------------------------------

####################################################################
#                 Identifying Extremes in pCO2                     #  
####################################################################
# Filter out rows where pCO2_max_value exceeds a chosen threshold 
high_pCO2 <- data[data$pCO2_max_value > 150, .SD, .SDcols = c(3:7, grep("pCO2_", colnames(data)), ncol(data)-1)]
# Order by pCO2_max_value in descending order to see the highest pCO2 values
setorder(high_pCO2, -pCO2_max_value)
# Print the number of rows in the filtered dataset for high pCO2 values
print(nrow(high_pCO2))
# Print the first few rows of the high pCO2 dataset to check the details
print(head(high_pCO2))

# Filter out rows where pCO2_min_value is less than a threshold 
low_pCO2 <- data[data$pCO2_min_value < 10, .SD, .SDcols = c(3:7, grep("pCO2_", colnames(data)), ncol(data)-1)]
# Order by pCO2_min_value in ascending order to see the lowest pCO2 values
setorder(low_pCO2, pCO2_min_value)
# Print the number of rows in the filtered dataset for low pCO2 values
print(nrow(low_pCO2))
# Print the first few rows of the low pCO2 dataset to check the details
print(head(low_pCO2))

# Clean up the temporary high and low pCO2 datasets to free memory
rm(high_pCO2, low_pCO2)

# ------------------------------------------------------------------------------

####################################################################
#                 Identifying Extremes in pO2                      #   
####################################################################
# Filter out rows where pO2_max_value exceeds a chosen threshold 
high_pO2 <- data[data$pO2_max_value > 500, .SD, .SDcols = c(3:7, grep("pO2_", colnames(data)), ncol(data)-1)]
# Order by pO2_max_value in descending order to see the highest pO2 values
setorder(high_pO2, -pO2_max_value)
# Print the number of rows in the filtered dataset for high pO2 values
print(nrow(high_pO2))
# Print the first few rows of the high pO2 dataset to check the details
print(head(high_pO2))

# Filter out rows where pO2_min_value is less than a threshold 
low_pO2 <- data[data$pO2_min_value < 10, .SD, .SDcols = c(3:7, grep("pO2_", colnames(data)), ncol(data)-1)]
# Order by pO2_min_value in ascending order to see the lowest pO2 values
setorder(low_pO2, pO2_min_value)
# Print the number of rows in the filtered dataset for low pO2 values
print(nrow(low_pO2))
# Print the first few rows of the low pO2 dataset to check the details
print(head(low_pO2))

# Clean up the temporary high and low pO2 datasets to free memory
rm(high_pO2, low_pO2)

#===============================================================================
# Relationships Between Variables
#===============================================================================

# compare the distribution of age for each discharge type
ggplot(data, aes(x = age, y = discharge_type)) +
  geom_boxplot(fill = "skyblue") + # Skyblue fill, dark blue borders
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Age Distribution by Discharge Type", x = "Age", y = "Discharge Type") +
  theme_minimal() # Adds a clean, minimal background

# compare the distribution of age for each diagnosis category
ggplot(data, aes(x = age, y = diagnosis_category)) + 
  geom_boxplot(fill = "skyblue") +
  theme(axis.text.y = element_text(angle = 0, hjust = 1, size = 8)) +  # Ensure readability
  labs(title = "Age Distribution by Diagnosis Category", x = "Age", y = "Diagnosis Category") +
  theme_minimal()

# Age distribution by diagnosis category and sex
ggplot(data, aes(x = age, y = diagnosis_category, fill = sex)) + 
  geom_boxplot(alpha=0.7) +
  theme(axis.text.y = element_text(angle = 0, hjust = 1, size = 8)) +  # Ensure readability
  labs(title = "Age Distribution by Diagnosis Category and Sex", 
       x = "Age", 
       y = "Diagnosis Category") +
  theme_minimal() +
  scale_fill_manual(values = c("coral", "skyblue")) +  # Custom colors for sexes (e.g., skyblue for male, pink for female)
  theme(
    axis.title.x = element_text(size = 16), # Adjusts the size of x-axis title
    axis.title.y = element_text(size = 16), # Adjusts the size of y-axis title
    axis.text.x = element_text(size = 14),  # Adjusts the size of x-axis text
    axis.text.y = element_text(size = 14)   # Adjusts the size of y-axis text
  )

# Age distribution by Discharge Type and sex
ggplot(data, aes(x = age, y = discharge_type, fill = sex)) + 
  geom_boxplot(alpha=0.7) +
  theme(axis.text.y = element_text(angle = 0, hjust = 1, size = 8)) +  # Ensure readability
  labs(title = "Age Distribution by Discharge Type and Sex", 
       x = "Age", 
       y = "Discharge Type") +
  theme_minimal() +
  scale_fill_manual(values = c("coral", "skyblue")) +  # Custom colors for sexes (e.g., skyblue for male, pink for female)
  theme(
    axis.title.x = element_text(size = 16), # Adjusts the size of x-axis title
    axis.title.y = element_text(size = 16), # Adjusts the size of y-axis title
    axis.text.x = element_text(size = 14),  # Adjusts the size of x-axis text
    axis.text.y = element_text(size = 14)   # Adjusts the size of y-axis text
  )
# ------------------------------------------------------------------------------

#  Length of Stay by Categories
ggplot(data, aes(x = length_of_stay_days, y = diagnosis_category)) +
  geom_boxplot(fill = "skyblue") +
  labs(title = "Length of Stay by Diagnosis Category", x = "Length of Stay (days)", y = "Diagnosis Category") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  theme_minimal()

# Faceted Boxplots (by Sex)
ggplot(data, aes(x = length_of_stay_days, y = discharge_type, fill = sex)) +
  geom_boxplot(alpha=0.7) +
  labs(title = "Length of Stay by Discharge Type and Sex", x = "Length of Stay (days)", y = "Discharge Type") +
  theme_minimal() +
  scale_fill_manual(values = c("coral", "skyblue")) +
  theme(
    axis.title.x = element_text(size = 16), # Adjusts the size of x-axis title
    axis.title.y = element_text(size = 16), # Adjusts the size of y-axis title
    axis.text.x = element_text(size = 14),  # Adjusts the size of x-axis text
    axis.text.y = element_text(size = 14)   # Adjusts the size of y-axis text
)

# Scatter Plot with Trend Line
ggplot(data, aes(x = age, y = length_of_stay_days)) +
  geom_point(color = "skyblue", alpha = 0.6) +
  geom_smooth(method = "lm", color = "darkblue") +
  labs(title = "Length of Stay vs. Age", x = "Age", y = "Length of Stay (days)") +
  theme_minimal()

#------------------------

# boxplot for Length of Stay by Discharge Type
ggplot(data, aes(x = length_of_stay_days, y = discharge_type)) +
  geom_boxplot(fill = "skyblue") +
  labs(title = "Length of Stay by Discharge Type", x = "Length of Stay (days)", y = "Discharge Type") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  theme_minimal()

# Calculate the average length of stay by discharge type
los_summary <- data %>%
  group_by(discharge_type) %>%
  summarize(
    avg_los = mean(length_of_stay_days, na.rm = TRUE),
    median_los = median(length_of_stay_days, na.rm = TRUE),
    count = n()
  ) %>%
  arrange(desc(count))

# Print the summary
print(los_summary)

# Filter the data for the "VISIT-HOSP" discharge type
visit_hosp_cases <- data %>%
  filter(discharge_type == "VISIT-HOSP")

# View the specific rows to check the length_of_stay_days
print(visit_hosp_cases[,1:7])

#=================================
#  LAB TEST TO FOCUS ON
#=================================
selected_lab_tests <- c("KA", "Leukn", "Hbn", "Eryn", "Hkn", "THZn", "MCHCn", "MCHn", "MCVn", "RDWn",
                          "MPVn","L", "I", "H", "NAT", "INRiH", "QUHD", "GL", "CR", "CRP", "NRBCmn", 
                          "EPIGFR", "Quicks", "ASAT", "ALAT", "GGT", "UREA", "CA", "CK", "BIg", "AP",
                          "NEUm.n", "LYMm.n", "EOSm.n", "MONm.n", "BASm.n", "IGm.n", "TNThsn", 
                          "H.Se", "I.Se", "L.Se", "LACT", "tHb", "CO.HB", "MTHB", "Tbga", "pH", "pCO2", "pO2")

length(selected_lab_tests)

# Generate column names for '_median' values
lab_test_median_columns <- paste0(selected_lab_tests, "_median_value")

# Subset the data.table
lab_test_data_median <- data[, ..lab_test_median_columns]

# Remove the "_median_value" suffix to keep just the lab test names
colnames(lab_test_data_median) <- sub("_median_value", "", colnames(lab_test_data_median))

# Calculate the correlation matrix
correlation_matrix <- cor(lab_test_data_median, use = "complete.obs")

# View the correlation matrix
print(correlation_matrix)

options(max.print = 10000)  # Adjust the number based on your matrix size
print(correlation_matrix)
# Visualize the correlation matrix with customized axis text size
ggcorrplot(correlation_matrix, 
           method = "square", 
           type = "lower", 
           lab = FALSE, 
           lab_size = 2, 
           title = "Correlation Matrix of Lab Test Median Values") +
  theme(axis.text.x = element_text(size = 11, angle = 90, hjust = 1),  # Adjust size and angle of X-axis labels
        axis.text.y = element_text(size = 11))                         # Adjust size of Y-axis labels


# Define the correlation thresholds for strong and moderate relationships
strong_threshold <- 0.7
moderate_threshold <- 0.4

# Extract the upper triangle of the matrix (without the diagonal)
upper_tri <- upper.tri(correlation_matrix)

# Get the values that are above the strong and moderate thresholds (positive and negative correlations)
strong_corr_pairs <- which(abs(correlation_matrix) > strong_threshold & upper_tri, arr.ind = TRUE)
moderate_corr_pairs <- which(abs(correlation_matrix) >= moderate_threshold & abs(correlation_matrix) <= strong_threshold & upper_tri, arr.ind = TRUE)

# Extract the variable names and correlation values for strong correlations
strong_corr_results <- data.frame(
  Var1 = rownames(correlation_matrix)[strong_corr_pairs[, 1]],
  Var2 = colnames(correlation_matrix)[strong_corr_pairs[, 2]],
  Correlation = correlation_matrix[strong_corr_pairs]
)

# Extract the variable names and correlation values for moderate correlations
moderate_corr_results <- data.frame(
  Var1 = rownames(correlation_matrix)[moderate_corr_pairs[, 1]],
  Var2 = colnames(correlation_matrix)[moderate_corr_pairs[, 2]],
  Correlation = correlation_matrix[moderate_corr_pairs]
)

# Strong correlations (|r| > 0.7)
cat("Strong correlations (|r| > 0.7):\n")

# Separate positive and negative correlations
strong_pos_corr <- strong_corr_results[strong_corr_results$Correlation > 0, ]
strong_neg_corr <- strong_corr_results[strong_corr_results$Correlation < 0, ]

# Sort by absolute value (strongest to weakest)
strong_pos_corr_sorted <- strong_pos_corr[order(-abs(strong_pos_corr$Correlation)), ]
strong_neg_corr_sorted <- strong_neg_corr[order(-abs(strong_neg_corr$Correlation)), ]

# Print sorted results
cat("\nStrong Positive Correlations (sorted by absolute value, strongest to weakest):\n")
print(strong_pos_corr_sorted)

cat("\nStrong Negative Correlations (sorted by absolute value, strongest to weakest):\n")
print(strong_neg_corr_sorted)

# Moderate correlations (0.4 <= |r| <= 0.7)
cat("\nModerate correlations (0.4 <= |r| <= 0.7):\n")

# Separate positive and negative correlations
moderate_pos_corr <- moderate_corr_results[moderate_corr_results$Correlation > 0, ]
moderate_neg_corr <- moderate_corr_results[moderate_corr_results$Correlation < 0, ]

# Sort by absolute value (strongest to weakest)
moderate_pos_corr_sorted <- moderate_pos_corr[order(-abs(moderate_pos_corr$Correlation)), ]
moderate_neg_corr_sorted <- moderate_neg_corr[order(-abs(moderate_neg_corr$Correlation)), ]

# Print sorted results
cat("\nModerate Positive Correlations (sorted by absolute value, strongest to weakest):\n")
print(moderate_pos_corr_sorted)

cat("\nModerate Negative Correlations (sorted by absolute value, strongest to weakest):\n")
print(moderate_neg_corr_sorted)

# -------------

# Correlation with LOS

# Combine the length_of_stay_days with the lab test mean data
lab_test_data_with_los <- cbind(data$length_of_stay_days, lab_test_data_median)
colnames(lab_test_data_with_los)[1] <- "length_of_stay_days"

# Calculate the correlation matrix for the combined data
correlation_with_los <- cor(lab_test_data_with_los, use = "complete.obs")

# View the correlations with length_of_stay_days
correlation_with_los["length_of_stay_days", ]

# Visualize the correlation matrix (including length_of_stay_days)
ggcorrplot(correlation_with_los, 
           method = "square", 
           type = "lower", 
           lab = FALSE, 
           lab_size = 3, 
           title = "Correlation Matrix with Length of Stay")

#####

# Extract the correlations with 'length_of_stay_days'
cor_with_los <- correlation_with_los["length_of_stay_days", -1]  # exclude the length_of_stay_days column itself

# Create a data frame to pass to ggcorrplot
cor_with_los_df <- data.frame(
  lab_test = names(cor_with_los),
  correlation = as.numeric(cor_with_los)
)

# Plot the correlations between lab tests and length_of_stay_days
ggplot(cor_with_los_df, aes(x = reorder(lab_test, correlation), y = correlation)) +
  geom_bar(stat = "identity", fill = "skyblue", alpha = 0.7) +
  coord_flip() +
  labs(
    title = "Correlation of Lab Tests with Length of Stay",
    x = "Lab Test",
    y = "Correlation with Length of Stay"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(size = 14), axis.text.y = element_text(size = 14),
        axis.title.x = element_text(size = 16), # Adjusts the size of x-axis title
        axis.title.y = element_text(size = 16)) # Adjusts the size of y-axis title)

#-------------------------------------------------------------------------------

# Heatmap (LOS across Categories)
library(viridis)
los_summary <- data %>%
  group_by(diagnosis_category, discharge_type) %>%
  summarize(mean_los = mean(length_of_stay_days, na.rm = TRUE))

ggplot(los_summary, aes(x = discharge_type, y = diagnosis_category, fill = mean_los)) +
  geom_tile() +
  scale_fill_viridis(option = "E", name = "Mean LOS") +
  labs(title = "Average Length of Stay by Diagnosis and Discharge Type",
       x = "Discharge Type", y = "Diagnosis Category") +
  theme_minimal()

# Group Age into Categories
data$age_group <- cut(data$age,
                      breaks = c(0, 18, 35, 50, 65, Inf),
                      labels = c("0-18", "19-35", "36-50", "51-65", "65+"),
                      right = FALSE)

# Boxplot of LOS by Age Group
ggplot(data, aes(x = age_group, y = length_of_stay_days)) +
  geom_boxplot(fill = "skyblue") +
  labs(title = "Length of Stay by Age Group", x = "Age Group", y = "Length of Stay (days)") +
  theme_minimal()

# Faceted Scatterplot by Age Group
ggplot(data, aes(x = length_of_stay_days, y = diagnosis_category, color = discharge_type)) +
  geom_point(alpha = 0.7) +
  facet_wrap(~ age_group) +
  labs(title = "Length of Stay by Diagnosis and Discharge Type (Faceted by Age Group)",
       x = "Length of Stay (days)",
       y = "Diagnosis Category") +
  theme_minimal()

#===============================================================================
# Summary of Script Actions
# This script has successfully:
# - Loaded and subsetted the lab data for selected test mean values.
# - Computed the correlation matrix between lab test mean values.
# - Visualized the correlation matrix of lab tests using a heatmap.
# - Analyzed the correlation between lab tests and length of stay (LOS).
# - Plotted the correlations of each lab test with LOS.
# - Generated a heatmap of LOS across diagnosis categories and discharge types.
# - Categorized age groups and visualized length of stay by age.
# - Created a faceted scatterplot showing the relationship between length of stay, 
#   diagnosis category, and discharge type by age group.
# Output Files:
#   - Correlation matrix plot (correlation_matrix.png)
#   - Correlation with LOS plot (correlation_with_los.png)
#   - LOS by Diagnosis and Discharge Type heatmap (LOS_heatmap.png)
#   - Length of Stay by Age Group boxplot (LOS_by_age_group.png)
#   - Faceted Scatterplot by Age Group (LOS_by_diagnosis_age_group.png)
#===============================================================================
