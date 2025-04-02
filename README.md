# Predicting hospital length of stay and discharge type using admission lab results, demographics, and diagnosis

Predicting hospital length of stay (LoS) and discharge type using admission clinical data and laboratory results. Accurate predictions could enhance patient care, optimize resource allocation, and facilitate timely coordination of social and medical services.

### Project Timeline:

```mermaid
gantt
    dateFormat  YYYY-MM-DD
    axisFormat %Y-%m-%w

    section Preparation
    Data cleaning in Python    :active, screv, 2025-02-24, 2025-03-24
    Data preprocessing         :active, screv, 2025-02-25, 2025-04-15
    Literature review          :active, screv, 2025-02-24, 2025-04-15

    Pragraph draft - State of the art             :milestone, 2025-04-15, 0d
    Abstract for Poster session application              :milestone, 2025-04-15, 0d

    section Implementation
    Modeling            :screv, 2025-04-15, 2025-06-15
    Paragraph draft - Methodology              :milestone, done, 2025-06-20, 0d
    Model validation    :crit,screv, 2025-06-01, 2025-07-15

    Poster presentation :milestone, done, 2025-07-02, 0d

    section Dissertation
    Theis writing :screv, 2025-07-15, 2025-08-15
```

### Ongoing Work:

- **Cleaning and preprocessing clinical and lab data** for predictive modeling of hospital length of stay and discharge type.
- **Investigating methods for handling missing data and outliers** in the dataset -> MissForest imputation algorithm and Isolation Forest for outlier detection.
- **Studying model selection, feature engineering, and variable selection techniques** for predicting hospital length of stay (LOS).
- **Reviewing literature** to understand related methods and approaches in predicting LOS and discharge type.

### Supervisor & Contributors

Supervisor: Prof. Alexander Benedikt Leichtle

MSc Student: Anna Scarpellini Pancrazi
