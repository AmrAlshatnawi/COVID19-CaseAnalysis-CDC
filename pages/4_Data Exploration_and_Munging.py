import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Covid Data Analysis - Data Exploration and Munging",
    page_icon="📊",
    #layout="wide"   
)
#st.markdown("""<style>body {zoom: 1;  /* Adjust this value as needed */}</style>""", unsafe_allow_html=True)

st.sidebar.markdown("""<div style="font-size: 17px;">✍️ <strong>Authors:</strong></div> 
\n&nbsp;                                  
<a href="https://www.linkedin.com/in/amralshatnawi/" style="display: inline-block; padding: 5px 7px; background-color: #871212; color: white; text-align: center; text-decoration: none; font-size: 15px; border-radius: 4px;">&nbsp;&nbsp;Amr Alshatnawi&nbsp;&nbsp;</a><br>             

<a href="https://www.linkedin.com/in/hailey-pangburn" style="display: inline-block; padding: 5px 7px; background-color: #871212; color: white; text-align: center; text-decoration: none; font-size: 15px; border-radius: 4px;">&nbsp;&nbsp;Hailey Pangburn&nbsp;&nbsp;</a><br>             
                    
<a href="mailto:mcmasters@uchicago.edu" style="display: inline-block; padding: 5px 7px; background-color: #871212; color: white; text-align: center; text-decoration: none; font-size: 15px; border-radius: 4px;">Richard McMasters</a><br>
""", unsafe_allow_html=True)

st.sidebar.write("---")
st.sidebar.markdown("""📅 March 9th, 2024""")
st.sidebar.image("Ulogo.png")


############################# start page content #############################

st.title("Data Exploration and Munging")
st.divider()


st.header("Sampling the Data")
st.markdown("""Given the extensive size of our dataset, containing 105 million records,
             we employed systematic sampling to generate a manageable sample dataset.
             This approach simplifies processing, analysis, and overall handling.
             We decided to select every 100th record from the original dataset for our sample.""")

with st.expander("👆 Expand to view systematic sampling code"):
    st.code("""
import pandas as pd

# data file path
file_path = 'drive/MyDrive/Data_Analysis/CDC_Covid_Data.csv'
# sample every 100th record
sampling_interval = 100  
chunk_size = 10000

# Placeholder for sampled rows
sampled_rows = []

# Open the file and iterate over it in chunks
with pd.read_csv(file_path, chunksize=chunk_size, na_values=['Missing', 'Unknown', 'NA', 'NaN']) as reader:
    for chunk_number, chunk in enumerate(reader):
        # Calculate the row index within the original file (global index) and select row
        start_row = chunk_number * chunk_size
        end_row = start_row + len(chunk)
        rows_to_sample = range(start_row, end_row, sampling_interval)

        # Adjust rows_to_sample to local indices within the chunk
        local_indices_to_sample = [row % chunk_size for row in rows_to_sample if row >= start_row and row < end_row]

        # Append the sampled rows from this chunk to the list
        sampled_rows.append(chunk.iloc[local_indices_to_sample])

# Concatenate all sampled rows into a single DataFrame
sampled_df = pd.concat(sampled_rows, ignore_index=True)

# file path to save the sampled DataFrame
output_file_path = 'systematic_sampled_covid_data_1M.csv'

# Save the DataFrame to a CSV file
sampled_df.to_csv(output_file_path, index=False)

            """)



st.header("Exploring the Data")

#data = pd.read_csv("systematic_sampled_covid_data.csv", na_values=['Missing', 'Unknown', 'NA', 'NaN', '', ' '])

# data 
csv_url = "https://www.dropbox.com/scl/fi/3ssnswv3158des6o0102v/systematic_sampled_covid_data_1M.csv?rlkey=kmf3ym19wdl3bq006fii5mcqa&st=8h53tkov&dl=1"
data = pd.read_csv(csv_url, na_values=['Missing', 'Unknown', 'NA', 'NaN', '', ' '])
st.write(f"The sampled dataset contains 1,045,441 rows and {data.shape[1]} columns.")
st.markdown("Data preview: ")
st.write(data.head())
with st.expander("👆 Expand to view code"):
    st.code("""
data = pd.read_csv("systematic_sampled_covid_data.csv", na_values=['Missing', 'Unknown', 'NA', 'NaN', '', ' '])
st.write(f"The sampled dataset contains {data.shape[0]} rows and {data.shape[1]} columns.")
print("Data preview: ")
data.head()
        """)

st.subheader("How have COVID-19 case counts varied over time since the onset of the pandemic?")


data['case_month'] = pd.to_datetime(data['case_month'], format='%Y-%m')

# Ensure dataframe is sorted by case_month
monthly_cases = data.groupby('case_month').size().reset_index(name='cases')
monthly_cases = monthly_cases.sort_values('case_month')

# Creating an interactive line plot with Plotly
fig = px.line(monthly_cases, x='case_month', y='cases', title='COVID-19 Cases Over Time',
              labels={'case_month': 'Month', 'cases': 'Number of Cases'},
              markers=True)

# Improve layout
fig.update_layout(xaxis_title='Date',
                  yaxis_title='Number of Cases',
                  #width=900,
                  #height=700,
                  xaxis=dict(rangeslider=dict(visible=True), type='date')) 
st.plotly_chart(fig)
st.markdown("""The plot illustrates the overall trend in COVID-19 case numbers from the pandemic's onset,
            highlighting a significant peak at the beginning of 2022. 
           This surge in cases is likely attributable to the emergence of the omicron variant,
            which was identified in November 2021. """)

with st.expander("👆 Expand to view plot code"):
    st.code("""
import plotly.express as px

data['case_month'] = pd.to_datetime(data['case_month'], format='%Y-%m')

# Ensure dataframe is sorted by case_month
monthly_cases = data.groupby('case_month').size().reset_index(name='cases')
monthly_cases = monthly_cases.sort_values('case_month')

# Creating an interactive line plot with Plotly
fig = px.line(monthly_cases, x='case_month', y='cases', title='COVID-19 Cases Over Time',
              labels={'case_month': 'Month', 'cases': 'Number of Cases'},
              markers=True)

# Improve layout
fig.update_layout(xaxis_title='Date',
                  yaxis_title='Number of Cases',
                  xaxis=dict(rangeslider=dict(visible=True), type='date')) 

fig.show()
""")
    

st.subheader("How do COVID-19 case counts vary among U.S. states?")
cases_per_state = data.groupby('res_state').size().reset_index(name='cases')


fig = px.choropleth(cases_per_state,
                    locations='res_state',
                    locationmode="USA-states",
                    color='cases',
                    scope="usa",
                    title='COVID-19 Cases by State',
                    hover_name='res_state',
                    hover_data={'res_state': False, 'cases': True}, 
                    color_continuous_scale=px.colors.sequential.YlOrRd, 
                    labels={'cases': 'Case Count'})  

# Enhance layout
fig.update_layout(
    title=dict(x=0.5),  
    geo=dict(
        lakecolor='rgb(255, 255, 255)', 
        showlakes=True,  # Show lakes
        landcolor='rgb(217, 217, 217)' 
    ),
    #width=900, 
    #height=700,
    margin=dict(t=50, l=0, r=0, b=0)
)

# Adjust color scale bar
fig.update_coloraxes(colorbar=dict(
    title='Total Cases',  
    thickness=20,  
    len=0.75, 
    bgcolor='rgba(255,255,255,0.5)',
    tickfont=dict(color='black'),  
    titlefont=dict(color='black')  
))


st.plotly_chart(fig)

st.markdown("""The choropleth map showcases the distribution of COVID-19 case counts, 
            highlighting that California has the highest number of cases,
             followed by Texas, New York, and Florida. This trend could be related to the
             larger population sizes in these states, considering they are the four most populated states in the U.S.""")

with st.expander("👆 Expand to view plot code"):
    st.code("""
import plotly.express as px

cases_per_state = data.groupby('res_state').size().reset_index(name='cases')


fig = px.choropleth(cases_per_state,
                    locations='res_state',
                    locationmode="USA-states",
                    color='cases',
                    scope="usa",
                    title='COVID-19 Cases by State',
                    hover_name='res_state',
                    hover_data={'res_state': False, 'cases': True}, 
                    color_continuous_scale=px.colors.sequential.YlOrRd, 
                    labels={'cases': 'Case Count'})  

# Enhance layout
fig.update_layout(
    title=dict(x=0.5),  
    geo=dict(
        lakecolor='rgb(255, 255, 255)', 
        showlakes=True,  # Show lakes
        landcolor='rgb(217, 217, 217)' 
    ),
    margin=dict(t=50, l=0, r=0, b=0)
)

# Adjust color scale bar
fig.update_coloraxes(colorbar=dict(
    title='Total Cases',  
    thickness=20,  
    len=0.75, 
    bgcolor='rgba(255,255,255,0.5)',
    tickfont=dict(color='black'),  
    titlefont=dict(color='black')  
))

fig.show()""")
        
st.subheader("How do COVID-19 case counts vary by gender?")

# Gender dataftame 
sex_counts = data['sex'].value_counts().reset_index()
sex_counts.columns = ['Sex', 'Count']  # Rename columns to match what we'll refer to in the plot

# Using the dataFrame in Plotly
fig = px.bar(sex_counts, x='Sex', y='Count', 
             title='Count of COVID-19 Cases by Sex',
             labels={'Count': 'Number of Cases'},
             color='Sex',
             color_discrete_map={'Female':'magenta', 'Male':'gold', 'Other':'green'})

st.plotly_chart(fig)

st.markdown("""The bar graph above illustrates the distribution of COVID-19 cases by sex, 
            revealing that **females** account for **544,048** cases, constituting **52%** of the total,
             while **males** represent **457,350** cases, making up **44%**. Additionally, there are **12** cases categorized as **'other'**,
             contributing to less than **1% closer to 0%** of the total, and **44,031** cases are marked as **missing** data,
             which comprises approximately **4%** of the overall case count.
""")

with st.expander("👆 Expand to view code"):
    st.code("""
import plotly.express as px

# Gender dataftame 
sex_counts = data['sex'].value_counts().reset_index()
sex_counts.columns = ['Sex', 'Count']  # Rename columns to match what we'll refer to in the plot

# Using the dataFrame in Plotly
fig = px.bar(sex_counts, x='Sex', y='Count', 
             title='Count of COVID-19 Cases by Sex',
             labels={'Count': 'Number of Cases'},
             color='Sex',
             color_discrete_map={'Female':'pink', 'Male':'blue', 'Other':'green'})
fig.show()
""")


st.subheader("How do COVID-19 case counts vary by Race?")
# Race dataftame 
race_counts = data['race'].value_counts().reset_index()
race_counts.columns = ['Race', 'Count']  # Rename columns to match what we'll refer to in the plot

# Using the dataFrame in Plotly
fig = px.bar(race_counts, x='Race', y='Count', 
             title='Count of COVID-19 Cases by Race',
             labels={'Count': 'Number of Cases'},
             color='Race',
             )
fig.update_layout(width=900, height=700)
st.plotly_chart(fig)

st.markdown("""The plot displayed above illustrates the distribution of COVID-19 cases by race,
indicating that the White population has a significantly higher number of cases relative to other racial groups.""")

with st.expander("👆 Expand to view code"):
    st.code("""
import plotly.express as px

# Race dataftame 
race_counts = data['race'].value_counts().reset_index()
race_counts.columns = ['Race', 'Count']  # Rename columns to match what we'll refer to in the plot

# Using the dataFrame in Plotly
fig = px.bar(race_counts, x='Race', y='Count', 
             title='Count of COVID-19 Cases by Race',
             labels={'Count': 'Number of Cases'},
             color='Race',
             )
fig.update_layout(width=900, height=700)
fig.show()
            
""")