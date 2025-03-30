# Notes LOS prediction

## Feature selection and data pre-processing

- source: https://www.nature.com/articles/s43856-024-00673-x#author-information

### 1. Diagnosis

- source: https://www.nature.com/articles/s43856-024-00673-x#author-information

"To produce features summarising expected length of stay (LOS) by each patient’s primary diagnosis, **we first summarised all primary diagnostic ICD-10 codes using Summary Hospital-level Mortality Indicator (SHMI) diagnosis groupings** [15]. Using data from all hospital admissions within the training dataset (01 February 2017 to 31 January 2019), we calculated as features the mean, standard deviation, median, maximum, and minimum of the LOS for each diagnostic category, to capture the effects of a current diagnostic category on future discharge probability16. We only used the training data to calculate the LOS characteristics of each diagnosis category, even when applying these estimates to the test dataset, to avoid possible data leakage, i.e., avoiding revealing information to the model that gives it an unrealistic advantage to make better predictions. ICD10 codes were assigned at discharge but were used as a proxy for the clinician’s working diagnosis (not available in our dataset) to inform model predictions in real time."

[15] https://digital.nhs.uk/data-and-information/publications/ci-hub/summary-hospital-level-mortality-indicator-shmi


### 2. Laboratory results

- source: https://www.nature.com/articles/s43856-024-00673-x#author-information

"For vital signs and laboratory tests, we used both numerical values reflecting the measurements themselves and the number of measurements within a particular time window, reflecting the fact that the decision to measure a vital sign or laboratory test is potentially informative in addition to the actual result obtained [17]. For example, clinicians may order additional laboratory measurements or record vital signs more frequently if patients are unstable18. To reduce collinearity, we grouped the number of measurements for vital signs (heart rate, respiratory rate, systolic blood pressure, diastolic blood pressure, **temperature**, oxygen saturation, O2 L/min, O2 delivery device, AVPU score, NEWS2 score), full blood counts (haemoglobin, haematocrit, mean cell volume, white cell count, platelets, neutrophils, lymphocytes, eosinophils, monocytes, basophils), renal function (creatinine, urea, potassium, sodium, estimated glomerular filtration rate (eGFR)), liver function (alkaline phosphatase, aspartate aminotransferase, alanine transaminase, albumin, bilirubin), bone profiles (adjusted calcium, magnesium, phosphate), clotting (activated partial thromboplastin time, prothrombin time), **blood gases (arterial, venous, or capillary combined, as labelling of blood gas type was not always complete/reliable)** (base excess, partial pressure of oxygen, partial pressure of carbon dioxide, lactate, arterial blood pH), and lipids (triglycerides, high-density lipoprotein cholesterol, total cholesterol, low-density lipoprotein cholesterol), respectively. The number of measurements for other blood tests were included individually.

We pre-processed the data by setting to missing implausible extreme values (Table below) not compatible with life, which typically represented uncorrectable errors in data entry, e.g., height 10 m, temperature 20 oC. 

| Test Name                               | Category | Lower Limit | Upper Limit | Unit                 |
|-----------------------------------------|----------|-------------|-------------|----------------------|
| heart rate                              | vital    | 30          | 300         | bpm                  |
| respiratory rate                        | vital    | 5           | 80          | cpm                  |
| temperature                             | vital    | 25          | 45          | °C                   |
| systolic blood pressure                 | vital    | 60          | 250         | mmHg                 |
| diastolic blood pressure                | vital    | 25          | 140         | mmHg                 |
| oxygen saturation                       | vital    | 70          | 100         | %                    |
| O2 L/min                                | vital    | 0           | 100         | L/min                |
| adjusted calcium                        | lab      | 1.5         | 3.6         | mmol/L               |
| albumin                                 | lab      | 9           | 55          | g/dL                 |
| alkaline phosphatase                    | lab      | 10          | 3500        | U/L                  |
| alanine transaminase                    | lab      | 2           | 4000        | U/L                  |
| amylase                                 | lab      | 2           | 4000        | U/L                  |
| activated partial thromboplastin time   | lab      | 12          | 250         | seconds              |
| aspartate aminotransferase              | lab      | 2           | 4000        | U/L                  |
| b12                                     | lab      | 50          | 2500        | pg/mL                |
| basophils                               | lab      | 0           | 1           | 10⁹/L                |
| base_excess                             | lab      | -30         | 25          | mEq/L                |
| lactate                                 | lab      | 0.1         | 25          | mmol/L               |
| pCo2                                    | lab      | 1.5         | 17          | kPa                  |
| pH                                      | lab      | 6.5         | 7.8         |                      |
| pO2                                     | lab      | 0.5         | 75          | kPa                  |
| bicarbonate                             | lab      | 4           | 40          | mmol/L               |
| bilirubin                               | lab      | 1           | 400         | µmol/L               |
| creatinine kinase                       | lab      | 5           | 150000      | U/L                  |
| creatinine                              | lab      | 8           | 5000        | mmol/L               |
| C-reactive protein                      | lab      | 0           | 600         | mg/L                 |
| d_dimer                                 | lab      | 50          | 100000      | mg/L                 |
| EGFR                                    | lab      | 0           | 150         | mL/min/1.73 m²       |
| eosinophils                             | lab      | 0           | 10          | 10⁹/L                |
| erythrocyte Sedimentation Rate          | lab      | 0           | 150         | mm/h                 |
| ferritin                                | lab      | 0.5         | 12000       | ng/mL                |
| folate                                  | lab      | 1.5         | 25          | ng/mL                |
| Gamma-glutamyl transferase              | lab      | 2           | 3000        | U/L                  |
| glucose                                 | lab      | 1           | 50          | mmol/L               |
| haemoglobin                             | lab      | 40          | 250         | g/L                  |
| hba1c                                   | lab      | 3           | 20          | %                    |
| haematocrit                             | lab      | 0.1         | 0.7         |                      |
| high-density lipoprotein cholesterol    | lab      | 0.2         | 4           | mmol/L               |
| iron                                    | lab      | 0.5         | 100         | µmol/L               |
| lactate dehydrogenase                   | lab      | 50          | 2000        | U/L                  |
| low-density lipoprotein cholesterol     | lab      | 0.1         | 10          | mmol/L               |
| lymphocytes                             | lab      | 0           | 150         | 10⁹/L                |
| Mean Cell Volume                        | lab      | 55          | 135         | fL                   |
| magnesium                               | lab      | 0.2         | 2.5         | mmol/L               |
| monocytes                               | lab      | 0           | 50          | 10⁹/L                |
| neutrophils                             | lab      | 0           | 150         | 10⁹/L                |
| phosphate                               | lab      | 0.1         | 6           | mmol/L               |
| platelets                               | lab      | 1           | 1200        | 10⁹/L                |
| potassium                               | lab      | 1.5         | 12          | mmol/L               |
| prostate-specific antigen               | lab      | 0           | 3000        | mg/L                 |
| prothrombin time                        | lab      | 8           | 200         | seconds              |
| sodium                                  | lab      | 100         | 180         | mmol/L               |
| total cholesterol                       | lab      | 1.2         | 20          | mmol/L               |
| total Ig                                | lab      | 0           | 20          | 10⁹/L                |
| transferrin                             | lab      | 0.2         | 10          | g/L                  |
| triglycerides                           | lab      | 0.2         | 50          | mmol/L               |
| troponin                                | lab      | 0           | 200         | mg/L                 |
| thyroid-stimulating hormone             | lab      | 0           | 250         | mU/L                 |
| urea                                    | lab      | 0.5         | 80          | mmol/L               |
| white cell count                        | lab      | 0           | 500         | 10⁹/L                |

**Categorical features were one-hot encoded.**

We did not perform data truncation and standardisation as decision trees-based algorithms are insensitive to the scale of the features, and we did not perform imputation of missing values because extreme gradient boosting (XGB) models can handle missing values by default using a ‘missing incorporated in attribute’ algorithm, which treats missing values as a separate category19,20. The proportion of missingness for features ranged from 0 to 88%, with the values of less commonly obtained laboratory tests exhibiting the greatest proportions of missingness (Supplementary Fig. 2).

-----------------------------------------------------------------------------------------------------------------------------------------------------------

# Notes DISCHARGE TYPE prediction

- source: https://doi.org/10.1186/s12913-022-08615-w (2022)

"The results of this study show XGBoost to be the most effective model for predicting between four common
discharge outcomes of ‘home’, ‘nursing facility’, ‘rehab’, and ‘death’, with 71% average c-statistic. The XGBoost model
was also the best-performer in the binary outcome experiment with a c-statistic of 76%."

-----------------------------------------------------------------------------------------------------------------------------------------------------------

# Predictive modeling

- source: Kuhn M, Johnson K. Applied predictive modeling., Vol. 26 Springer; 2013.

"There are a number of common reasons why predictive models fail, and we
address each of these in subsequent chapters. The common culprits include (1)
inadequate pre-processing of the data, (2) inadequate model validation, (3)
unjustiﬁed extrapolation (e.g., application of the model to data that reside in
a space which the model has never seen), or, most importantly, (4) over-ﬁtting
the model to the existing data.

Furthermore, predictive modelers often only
explore relatively few models when searching for predictive relationships. This
is usually due to either modelers’ preference for, knowledge of, or expertise
in, only a few models or the lack of available software that would enable them
to explore a wide range of techniques."

"1.1 Prediction Versus Interpretation

For the examples listed above, historical data likely exist that can be used
to create a mathematical tool to predict future, unseen cases. Furthermore,
the foremost objective of these examples is not to understand why something
will (or will not) occur. Instead, we are primarily interested in accurately pro-
jecting the chances that something will (or will not) happen. Notice that the
focus of this type of modeling is to optimize prediction accuracy. [...]
The tension between prediction and interpretation is also present in the
medical ﬁeld. [...] If a model is created to make this
prediction, it should not be constrained by the requirement of interpretability.
A strong argument could be made that this would be unethical. As long as
the model can be appropriately validated, it should not matter whether it is
a black box or a simple, interpretable model.
While the primary interest of predictive modeling is to generate accurate
predictions, a secondary interest may be to interpret the model and under-
stand why it works. **The unfortunate reality is that as we push towards higher
accuracy, models become more complex and their interpretability becomes
more diﬃcult. This is almost always the trade-oﬀ we make when prediction
accuracy is the primary goal.**"

"To summarize, the foundation of an eﬀective predictive model is laid with
intuition and deep knowledge of the problem context, which are entirely vi-
tal for driving decisions about model development. That process begins with
relevant data, another key ingredient. The third ingredient is a versatile com-
putational toolbox which includes techniques for data pre-processing and vi-
sualization as well as a suite of modeling tools for handling a range of possible
scenarios such as those that are described in Table 1.1."

"2 A Short Tour of the Predictive Modeling Process

[...] there
is no single model that will always do better than any other model. Because
of this, a strong case can be made to try a wide variety of techniques, then
determine which model to focus on. In our example, a simple plot of the
data shows that there is a nonlinear relationship between the outcome and
the predictor. Given this knowledge, we might exclude linear models from
consideration, but there is still a wide variety of techniques to evaluate.

At face value, model building appears straightforward: pick a modeling tech-
nique, plug in data, and generate a prediction. While this approach will gener-
ate a predictive model, it will most likely not generate a reliable, trustworthy
model for predicting new samples. To get this type of model, we must ﬁrst
understand the data and the objective of the modeling. Upon understand-
ing the data and objectives, we then pre-process and split the data. Only
after these steps do we ﬁnally proceed to building, evaluating, and selecting
models."

"As with many questions of statistics, the answer to “which feature engi-
neering methods are the best?” is that it depends. Speciﬁcally, it depends on
the model being used and the true relationship with the outcome."