import streamlit as st
import pandas as pd
from streamlit.components.v1 import html

st.set_page_config(
    page_title="Covid Data Analysis - Data Source",
    page_icon="📊",
    #layout="wide"   
)
st.markdown("""<style>body {zoom: 1.4;  /* Adjust this value as needed */}</style>""", unsafe_allow_html=True)

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


st.title("Data Source and Description")
st.divider()

st.markdown("""
In our analysis, we utilized the CDC's *COVID-19 Case Surveillance Public Use Data with Geography* dataset,
which is regularly updated. The version we accessed was last updated on February 20, 2024.
This comprehensive dataset comprises approximately **105 million** rows, each representing a de-identified patient case.
It features **19 columns**, encompassing a wide range of information including demographic details, geographical data,
and related elements. Detailed descriptions of the columns are provided below.
""")

columns = {
    'Column Name': ['case_month','res_state', 'state_fips_code', 'res_county', 'county_fips_code', 'age_group', 'sex', 
                    'race','ethnicity', 'case_positive_specimen_interval', 'case_onset_interval', 'process', 'exposure_yn',
                    'current_status', 'symptom_status', 'hosp_yn', 'icu_yn','death_yn', 'underlying_conditions_yn'],
    'Description': ['The earlier of month the Clinical Date (date related to the illness or specimen collection) or the Date Received by CDC',
                    'State of residence','State FIPS code','County of residence','County FIPS code','Age group [0 - 17 years; 18 - 49 years; 50 - 64 years; 65 + years; Unknown; Missing; NA, if value suppressed for privacy protection.]',
                    'Sex [Female; Male; Other; Unknown; Missing; NA, if value suppressed for privacy protection.]','Race [American Indian/Alaska Native; Asian; Black; Multiple/Other; Native Hawaiian/Other Pacific Islander; White; Unknown; Missing; NA, if value suppressed for privacy protection.]',
                    'Ethnicity [Hispanic; Non-Hispanic; Unknown; Missing; NA, if value suppressed for privacy protection.]',
                    'Weeks between earliest date and date of first positive specimen collection','Weeks between earliest date and date of symptom onset.',
                    'Under what process was the case first identified? [Clinical evaluation; Routine surveillance; Contact tracing of case patient; Multiple; Other; Unknown; Missing]',
                    'In the 14 days prior to illness onset, did the patient have any of the following known exposures: domestic travel, international travel, cruise ship or vessel travel as a passenger or crew member, workplace, airport/airplane, adult congregate living facility (nursing, assisted living, or long-term care facility), school/university/childcare center, correctional facility, community event/mass gathering, animal with confirmed or suspected COVID-19, other exposure, contact with a known COVID-19 case? [Yes, Unknown, Missing]',
                    'What is the current status of this person? [Laboratory-confirmed case, Probable case]',
                    'What is the symptom status of this person? [Asymptomatic, Symptomatic, Unknown, Missing]',
                    'Was the patient hospitalized? [Yes, No, Unknown, Missing]','Was the patient admitted to an intensive care unit (ICU)? [Yes, No, Unknown, Missing]',
                    'Did the patient die as a result of this illness? [Yes; No; Unknown; Missing; NA, if value suppressed for privacy protection.]',
                    'Did the patient have one or more of the underlying medical conditions and risk behaviors: diabetes mellitus, hypertension, severe obesity (BMI>40), cardiovascular disease, chronic renal disease, chronic liver disease, chronic lung disease, other chronic diseases, immunosuppressive condition, autoimmune condition, current smoker, former smoker, substance abuse or misuse, disability, psychological/psychiatric, pregnancy, other. [Yes, No, blank]']
    
}

with st.expander("👆 Click to see columns description"):
    columns_df = pd.DataFrame(columns)
    st.table(columns_df)

st.write("")
st.write("")

URL_CDC = "https://data.cdc.gov/Case-Surveillance/COVID-19-Case-Surveillance-Public-Use-Data-with-Ge/n8mc-b4w4/about_data"

st.markdown(
    f'<a href="{URL_CDC}" style="display: inline-block; padding: 12px 20px; background-color: #085492; color: white; text-align: center; text-decoration: none; font-size: 16px; border-radius: 4px;">**Click here to access the dataset on CDC website**</a>',
    unsafe_allow_html=True
)

st.image("./cdc.png")