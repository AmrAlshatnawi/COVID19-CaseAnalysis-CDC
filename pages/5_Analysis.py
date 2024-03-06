import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.stats import chisquare
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Covid Data Analysis - Analysis",
    page_icon="📊",
    #layout="wide"   
)
st.markdown("""<style>body {zoom: 1.4;  /* Adjust this value as needed */}</style>""", unsafe_allow_html=True)

st.sidebar.markdown("""✍️**Authors:**                 
    Amr Alshatnawi       
    Hailey Pangburn                 
    Richard McMasters""")
st.sidebar.write("---")
st.sidebar.markdown("""📅 March 9th, 2024""")
st.sidebar.image("Ulogo.png")

############################# start page content #############################

st.title("Analysis")
st.divider()

data = pd.read_csv("systematic_sampled_covid_data.csv", na_values=['Missing', 'Unknown', 'NA', 'NaN', '', ' '])

st.header("""Is there a significant difference in COVID-19 case counts between different age groups?""")

st.markdown("""To address the question of whether there is a significant difference in COVID-19 case counts between different
               age groups, relative to the population size of those groups, we examined the distribution of cases across
               each age category. Initially, we hypothesized that the distribution of cases would be proportional to the
               population sizes of the various age groups. To lay the groundwork for our analysis,
               we started by calculating the case counts for each age group.""")

case_counts_by_age_group = data['age_group'].value_counts().sort_index()
df_age_case = case_counts_by_age_group.reset_index()

df_age_case.columns = ['Age_Group', 'Count_of_Cases']

st.dataframe(df_age_case.head())

with st.expander("👆 Expand to view code"):
    st.code("""
data = pd.read_csv("systematic_sampled_covid_data.csv", na_values=['Missing', 'Unknown', 'NA', 'NaN', '', ' '])
case_counts_by_age_group = data['age_group'].value_counts().sort_index()
df_age_case = case_counts_by_age_group.reset_index()

df_age_case.columns = ['Age_Group', 'Count_of_Cases']

df_age_case.head()
""")
    
# plot the results
fig = px.bar(df_age_case, x='Age_Group', y='Count_of_Cases',
             labels={'Age_Group': 'Age Group', 'Count_of_Cases': 'Number of Cases'},
             title='Summary of COVID-19 Cases by Age Group', 
             color='Age_Group')
fig.update_xaxes(tickangle=45)

st.plotly_chart(fig)

with st.expander("👆 Expand to view code"):
    st.code("""
# plot the results
fig = px.bar(df_age_case, x='Age_Group', y='Count_of_Cases',
             labels={'Age_Group': 'Age Group', 'Count_of_Cases': 'Number of Cases'},
             title='Summary of COVID-19 Cases by Age Group')
fig.update_xaxes(tickangle=45)
fig.show()

""")
    
st.markdown("""Observing the bar graph above reveals that the distribution of cases across most age groups is relatively similar,
                with a notable exception. The 18 to 49 age group exhibits a significant discrepancy,
                displaying a noticeably higher number of cases in comparison to the other groups.
            """)

st.markdown("""To determine if these observed differences are statistically significant,
                we utilized the Chi-square goodness-of-fit test. This test allowed us to compare the actual
                distribution of cases across age groups against the expected distribution based on the population
                sizes of each age group.
            """)

st.subheader("chi square goodness of fit test results")
#########################################################################
# Population sizes for each age group
populations = {'0-17 years': 74000000, '18-49 years': 132000000, '50-64 years': 64000000, '65+ years': 53000000}

# Observed COVID-19 cases
observed_cases = {'0-17 years': 171498, '18-49 years': 518457, '50-64 years': 188827, '65+ years': 145737}

total_population = sum(populations.values())
total_cases = sum(observed_cases.values())

expected_cases = {}
for group in populations:
    proportion_of_population = populations[group] / total_population
    expected_cases[group] = proportion_of_population * total_cases

# Convert observed and expected cases to lists
observed = list(observed_cases.values())
expected = list(expected_cases.values())

# Perform the Chi-square goodness-of-fit test
chi2_stat, p_value = chisquare(observed, f_exp=expected)


# create dataframe to show test results 
results = {
    'P-Value': [p_value],
    'Chi-square_Statistic': [chi2_stat]
}
results_df = pd.DataFrame(results)

# Format the p-value 
results_df['P-Value'] = results_df['P-Value'].apply(lambda x: f"{x:.4f}")

st.dataframe(results_df)

st.markdown("""With the p-value being less than 0.05, we reject the null hypothesis.
               This low p-value indicates that the actual distribution of COVID-19 cases across age groups does not align
               with the distribution expected from the population sizes of these groups, marking the observed differences
               in distribution as statistically significant. Such variations could stem from numerous factors,
               including but not limited to, differences in social behaviors, types of employment, or varying levels
               of exposure risk among the age groups.
""")
st.success("H1: The distribution of COVID-19 cases across age groups is not proportional to the population distribution of those age groups, suggesting that, relative to their population size, certain age groups are more likely to contract COVID-19 than others.")

with st.expander("👆 Expand to view code"):
    st.code("""
from scipy.stats import chisquare

# Population sizes for each age group
populations = {'0-17 years': 74000000, '18-49 years': 132000000, '50-64 years': 64000000, '65+ years': 53000000}

# Observed COVID-19 cases
observed_cases = {'0-17 years': 171498, '18-49 years': 518457, '50-64 years': 188827, '65+ years': 145737}

total_population = sum(populations.values())
total_cases = sum(observed_cases.values())

expected_cases = {}
for group in populations:
    proportion_of_population = populations[group] / total_population
    expected_cases[group] = proportion_of_population * total_cases

# Convert observed and expected cases to lists in the same order
observed = list(observed_cases.values())
expected = list(expected_cases.values())

# Perform the Chi-square goodness-of-fit test
chi2_stat, p_value = chisquare(observed, f_exp=expected)


# create dataframe to show test results 
results = {
    'P-Value': [p_value],
    'Chi-square_Statistic': [chi2_stat]
}
results_df = pd.DataFrame(results)

# Format the p-value 
results_df['P-Value'] = results_df['P-Value'].apply(lambda x: f"{x:.4f}")
            
results_df.head()
""")
    
st.divider()

################################################## Start logistic Regression ##################################################

st.header("Do gender, age group, and case year significantly associate with COVID-19 mortality outcomes?")


data_LR = pd.read_csv("systematic_sampled_covid_data.csv", na_values=['Missing', 'Unknown', 'NA', 'NaN','Other'])

# List of columns to drop
columns_to_drop = ['res_county', 'res_state', 'current_status', 'state_fips_code', 'county_fips_code',
                    'process', 'exposure_yn', 'symptom_status', 'hosp_yn', 'icu_yn', 'ethnicity',
                      'underlying_conditions_yn','case_positive_specimen_interval', 'case_onset_interval', 'race']

# Drop columns not need for Logistic Regression
data_LR = data_LR.drop(columns=columns_to_drop, axis=1)

# Replace 'Missing', 'Unknown', 'NA', 'NaN' with NaN to standardize missing values
data_LR.replace(['Missing', 'Unknown', 'NA', 'NaN', '', ' '], pd.NA, inplace=True)

# Dropping rows with missing values in any column
data_LR = data_LR.dropna()


# Grouping age
data_LR['age_group'] = data_LR['age_group'].replace({
    '0 - 17 years': '0 - 64 years',
    '18 to 49 years': '0 - 64 years',
})

# Ensure 'death_yn' is numeric
data_LR['death_yn'] = data_LR['death_yn'].map({'Yes': 1, 'No': 0})


# Convert case_month to datetime and extract useful features
data_LR['case_month'] = pd.to_datetime(data_LR['case_month'], format='%Y-%m')
data_LR['year'] = data_LR['case_month'].dt.year
data_LR['year'] = data_LR['year'].astype(str)

# Group 2023 and 2024 together
data_LR['year'].replace({'2022': '2022 or later', '2023': '2022 or later', '2024': '2022 or later'}, inplace=True)

# Creating dummy variables
data_LR = pd.get_dummies(data_LR, columns=['sex', 'age_group', 'year'], drop_first=True)

st.markdown("""
To address our research question, we chose logistic regression as our method of analysis.
Our intention was to focus exclusively on a predefined set of variables: gender, age group, and case year, for our model.
Thus, we initiated our process by refining our dataset to solely encompass these selected predictors along with the
dependent variable, which is mortality. We streamlined the age categories into three distinct groups: 0-49 years, 50-64 years,
and 65 years and above, for clarity and simplicity. Furthermore, we adapted the case month data to extract only the year,
consolidating all cases from 2022 onwards into a single category. This approach resulted in our analysis spanning the years
2020, 2021, and "2022 or later." In preparation for logistic regression, we transformed our predictors into dummy variables.
This diligent data preparation reduced our dataset to **306,929 rows**. Below is head of the dataframe showcasing our data structure. 
""")

st.dataframe(data_LR.drop(['case_month'], axis=1).head())

with st.expander("👆 Expand to view code"):
    st.code("""
data_LR = pd.read_csv("systematic_sampled_covid_data.csv", na_values=['Missing', 'Unknown', 'NA', 'NaN','Other'])

# List of columns to drop
columns_to_drop = ['res_county', 'res_state', 'current_status', 'state_fips_code', 'county_fips_code',
                    'process', 'exposure_yn', 'symptom_status', 'hosp_yn', 'icu_yn', 'ethnicity',
                      'underlying_conditions_yn','case_positive_specimen_interval', 'case_onset_interval', 'race']

# Drop columns not need for Logistic Regression
data_LR = data_LR.drop(columns=columns_to_drop, axis=1)

# Replace 'Missing', 'Unknown', 'NA', 'NaN' with NaN to standardize missing values
data_LR.replace(['Missing', 'Unknown', 'NA', 'NaN', '', ' '], pd.NA, inplace=True)

# Dropping rows with missing values in any column
data_LR = data_LR.dropna()


# Grouping age
data_LR['age_group'] = data_LR['age_group'].replace({
    '0 - 17 years': '0 - 64 years',
    '18 to 49 years': '0 - 64 years',
})

# Ensure 'death_yn' is numeric
data_LR['death_yn'] = data_LR['death_yn'].map({'Yes': 1, 'No': 0})


# Convert case_month to datetime and extract useful features
data_LR['case_month'] = pd.to_datetime(data_LR['case_month'], format='%Y-%m')
data_LR['year'] = data_LR['case_month'].dt.year
data_LR['year'] = data_LR['year'].astype(str)

# Group 2023 and 2024 together
data_LR['year'].replace({'2022': '2022 or later', '2023': '2022 or later', '2024': '2022 or later'}, inplace=True)

# Creating dummy variables
data_LR = pd.get_dummies(data_LR, columns=['sex', 'age_group', 'year'], drop_first=True)
""")
    
st.markdown("""
Once our data was prepared, we proceeded to divide it into training and testing subsets,
allocating **20%** for testing and the remaining **80%** for training purposes. We observed an imbalance in our dataset,
with the minority class comprising only **4,702 rows (1.5%)** and the majority class accounting for **302,227 rows (98.5%)**.
Despite this imbalance, we chose to proceed with our logistic regression analysis to assess its impact on the statistical
outcomes.
""")

# Getting the data ready for splitting
X = data_LR.drop(['death_yn', 'case_month'], axis=1)  
y = data_LR['death_yn']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


st.subheader("Logistic Regression results using imbalanced data")
########################## using imbalanced data ##########################
# Adding a constant to the model for the intercept
X_train_sm = sm.add_constant(X_train)

X_train_sm = X_train_sm.astype(float)

# Fit the logistic regression model
logit_model = sm.Logit(y_train, X_train_sm)
result = logit_model.fit()

st.write(result.summary())

with st.expander("👆 Expand to view code"):
    st.code("""
# Getting the data ready for splitting
X = data_LR.drop(['death_yn', 'case_month'], axis=1)  
y = data_LR['death_yn']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

########################## using imbalanced data ##########################
# Adding a constant to the model for the intercept
X_train_sm = sm.add_constant(X_train)

X_train_sm = X_train_sm.astype(float)

# Fit the logistic regression model
logit_model = sm.Logit(y_train, X_train_sm)
result = logit_model.fit()

print(result.summary())

""")
    
st.markdown("""
In our logistic regression model, all predictor variables are statistically significant, with **p-values of less than 0.05**.
In addition, the model's Likelihood Ratio **(LLR) p-value is also 0**, indicating that overall, the model is statistically significant.
These findings affirm the robustness and relevance of our predictors in relation to the outcome variable.
""")

st.markdown("""
However, training a model on imbalanced data can lead to several challenges,
including bias towards the majority class, misleading accuracy metrics, and potential overfitting to the majority class.
To address these concerns, we implemented a resampling strategy to balance the dataset more effectively. Initially,
we undersampled the majority class using a **RandomUnderSampler**, targeting a ratio of 0.1. This ratio aimed to adjust the
dataset so that the minority class would represent approximately **10%** of the majority class's size. To further balance the
dataset, we employed the Synthetic Minority Over-sampling Technique **(SMOTE)** for oversampling the minority class. With a
sampling ratio of **0.5**, our goal was to increase the minority class size to half that of the majority class post-oversampling.
This two-step resampling process was designed to mitigate the risks associated with imbalanced datasets and improve
the model's overall performance. Then we utilized the resampled data to investigate its impact on the outcomes
of the logistic regression analysis.
""")

####################################

# creating resampled dataset to adjust for data imbalance
resampling_strategy = Pipeline([
    ('undersample', RandomUnderSampler(sampling_strategy=0.1)),  
    ('oversample', SMOTE(sampling_strategy=0.5))
])

# resampled dataset for training
x_resampled, y_resampled = resampling_strategy.fit_resample(X_train.astype(float), y_train)


########################## using Undersampled and Oversampled data ##########################

st.subheader("Logistic Regression results using Undersampled and Oversampled data")

# Adding a constant to the model for the intercept
X_train_sm_1 = sm.add_constant(x_resampled)

X_train_sm_1 = X_train_sm_1.astype(float)

# Fit the logistic regression model
logit_model_1 = sm.Logit(y_resampled, X_train_sm_1)
result_1 = logit_model_1.fit()

# Print the summary of the regression
st.write(result_1.summary())

with st.expander("👆 Expand to view code"):
    st.code("""

# creating resampled dataset to adjust for data imbalance
resampling_strategy = Pipeline([
    ('undersample', RandomUnderSampler(sampling_strategy=0.1)),  
    ('oversample', SMOTE(sampling_strategy=0.5))
])

# resampled dataset for training
x_resampled, y_resampled = resampling_strategy.fit_resample(X_train.astype(float), y_train)


########################## using Undersampled and Oversampled data ##########################


# Adding a constant to the model for the intercept
X_train_sm_1 = sm.add_constant(x_resampled)

X_train_sm_1 = X_train_sm_1.astype(float)

# Fit the logistic regression model
logit_model_1 = sm.Logit(y_resampled, X_train_sm_1)
result_1 = logit_model_1.fit()

# Print the summary of the regression
print(result_1.summary())

""")
    
st.markdown("""
Upon applying the resampled data to our logistic regression model, we observed minimal changes in the results.
All predictors retained their significance, showing coefficients that were very similar to those obtained with the
original dataset. Also, the model retained its overall significance. A notable change, however, was observed in the
intercept's (constant) coefficient, which shifted from **-7.1371 to -3.8469** in terms of log odds. This shift suggests that,
within the context of a more balanced dataset, the baseline probability of death when controlling all other variables
is higher compared to what was noted with the imbalanced dataset. Given these results, we reject the null hypothesis and
find significant evidence to suggest that gender, age group, and case year significantly predict COVID-19 mortality.
""")

st.success("H1: Gender, age group, and case year significantly predict COVID-19 mortality.")

st.write("")
########################## fit the model and predict on test set ##########################

st.markdown("Next, we proceeded to fit the model using the resampled data, and predict on the test dataset.")
# Initialize the logistic regression model
logreg = LogisticRegression(solver='liblinear', random_state=42)

# Fit the model on the training data
logreg.fit(x_resampled, y_resampled)

# predict on the original unmodified test set
y_pred = logreg.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
# Extracting True Negatives, False Positives, False Negatives, and True Positives
tn, fp, fn, tp = cm.ravel()

# Calculating metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
Sensitivity = recall_score(y_test, y_pred) 
specificity = tn / (tn + fp)
f1 = f1_score(y_test, y_pred)


test_results = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Sensitivity (Recall)', 'Specificity', 'F1 Score'],
    'Value': [accuracy, precision, Sensitivity, specificity, f1]
})

st.dataframe(test_results)

st.markdown("""

The confusion matrix table above highlights the model's performance, showcasing an overall accuracy of **89.94%**, effectively predicting
true positives and negatives. However, its precision at **10.62%** reveals a significant proportion of false positives. The model demonstrates a strong ability to detect actual deaths, with a sensitivity
of **81.44%**, and accurately identifies **90.07%** of true negatives, showcasing its effectiveness in recognizing survivors of
COVID-19. Despite these strengths, a low F1 score of **0.1879** indicates a challenge in achieving a balance between precision
and sensitivity, reflecting the model's tendency to favor sensitivity at the expense of higher false positive rates.
""")
with st.expander("👆 Expand to view code"):
    st.code("""
########################## fit the model and predict on test set ##########################
# Initialize the logistic regression model
logreg = LogisticRegression(solver='liblinear', random_state=42)

# Fit the model on the training data
logreg.fit(x_resampled, y_resampled)

# predict on the original unmodified test set
y_pred = logreg.predict(X_test)


cm = confusion_matrix(y_test, y_pred)
# Extracting True Negatives, False Positives, False Negatives, and True Positives
tn, fp, fn, tp = cm.ravel()

# Calculating metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
Sensitivity = recall_score(y_test, y_pred) 
specificity = tn / (tn + fp)
f1 = f1_score(y_test, y_pred)


test_results = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Sensitivity (Recall)', 'Specificity', 'F1 Score'],
    'Value': [accuracy, precision, Sensitivity, specificity, f1]
})

test_results.head()
""")

########################## Plotting confusion matrix ##########################

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Death', 'Death'])
disp.plot(cmap='Reds')
plt.title('Confusion Matrix for COVID-19 Mortality Prediction')
st.pyplot(plt)

st.markdown("""
Displayed above is the confusion matrix, offering a visual representation of True Positives (TP),
True Negatives (TN), False Positives (FP), and False Negatives (FN).
""")
with st.expander("👆 Expand to view code"):
    st.code("""
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Death', 'Death'])
disp.plot(cmap='Reds')
plt.title('Confusion Matrix for COVID-19 Mortality Prediction')
plt.show()
""")