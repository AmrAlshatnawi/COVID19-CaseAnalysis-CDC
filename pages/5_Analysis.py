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

st.subheader("""Is there a significant difference in COVID-19 case counts between different age groups?""")

st.markdown("""To address the question of whether there is a significant difference in COVID-19 case counts between different age groups,
             we examined the distribution of cases across each age category. Prior to conducting our analysis,
             we hypothesized that the distribution would be uniform across the various age groups.
             To enhance our understanding and facilitate a thorough analysis, 
             we began by calculating the case counts for each age group.""")

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
                distribution of cases across age groups against our initial prediction, which was based on the
                expectation of a uniform distribution of COVID-19 cases across all age groups. 
            """)

st.subheader("chi square goodness of fit test results")



# Assuming an even distribution, each group's expected count is the total count divided by the number of groups
total_cases = data['age_group'].count()
expected_count_per_group = total_cases / len(case_counts_by_age_group)
expected_counts = [expected_count_per_group] * len(case_counts_by_age_group)

# Perform the test
chi2_stat, p_value = chisquare(f_obs=case_counts_by_age_group, f_exp=expected_counts)

# create dataframe to show test results 
results = {
    'P-Value': [p_value],
    'Chi-square_Statistic': [chi2_stat]
}
results_df = pd.DataFrame(results)

# Format the p-value 
results_df['P-Value'] = results_df['P-Value'].apply(lambda x: f"{x:.4f}")

st.dataframe(results_df)

st.markdown("""Given that the p-value is less than 0.05, we reject the null hypothesis.
               This low p-value suggests that the distribution of COVID-19 cases is not uniform across age groups,
               and the observed difference in distribution is statistically significant. This difference could be due
               to various factors such as differences in social behaviors, employment types, or exposure risks associated with different age groups. 
""")
st.error("H0: The distribution of COVID-19 cases is uniform across age groups, indicating that age does not influence the likelihood of contracting COVID-19.")
st.success("H1: The distribution of COVID-19 cases is not uniform across age groups, suggesting that certain age groups are more likely to contract COVID-19 than others.")

with st.expander("👆 Expand to view code"):
    st.code("""
from scipy.stats import chisquare

# Assuming an even distribution, each group's expected count is the total count divided by the number of groups
total_cases = data['age_group'].count()
expected_count_per_group = total_cases / len(case_counts_by_age_group)
expected_counts = [expected_count_per_group] * len(case_counts_by_age_group)

# Perform the test
chi2_stat, p_value = chisquare(f_obs=case_counts_by_age_group, f_exp=expected_counts)

# create dataframe to show test results 
results = {
    'P-Value': {p_value},
    'Chi-square_Statistic': {chi2_stat}
}
results_df = pd.DataFrame(results)

# Format the p-value 
results_df['P-Value'] = results_df['P-Value'].apply(lambda x: f"{x:.4f}")

results_df.head()

""")
    
st.divider()

################################################## Start logistic Regression ##################################################

st.subheader("Do gender, age group, and case year significantly associate with COVID-19 mortality outcomes?")

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

# Getting the data ready for splitting
X = data_LR.drop(['death_yn', 'case_month'], axis=1)  
y = data_LR['death_yn']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# creating resampled dataset to adjust for data imbalance
resampling_strategy = Pipeline([
    ('undersample', RandomUnderSampler(sampling_strategy=0.1)),  
    ('oversample', SMOTE(sampling_strategy=0.5))
])

# resampled dataset for training
x_resampled, y_resampled = resampling_strategy.fit_resample(X_train.astype(float), y_train)

st.subheader("Logistic Regression results using imbalanced data")

########################## using imbalanced data ##########################
# Adding a constant to the model for the intercept
X_train_sm = sm.add_constant(X_train)

X_train_sm = X_train_sm.astype(float)

# Fit the logistic regression model
logit_model = sm.Logit(y_train, X_train_sm)
result = logit_model.fit()

st.write(result.summary())


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

########################## fit the model and predict on test set ##########################

st.markdown("Results from test")
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

########################## Plotting confusion matrix ##########################

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Death', 'Death'])
disp.plot(cmap='Reds')
plt.title('Confusion Matrix for COVID-19 Mortality Prediction')
st.pyplot(plt)





