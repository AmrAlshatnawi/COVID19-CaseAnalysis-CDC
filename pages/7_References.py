import streamlit as st

st.set_page_config(
    page_title="Covid Data Analysis - References",
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

# def add_side_title():
#     st.markdown(
#         """
#         <style>
#             [data-testid="stSidebarNav"]::before {
#                 content:"MSBI 32000 Winter 2024";
#                 margin-left: 20px;
#                 margin-top: 20px;
#                 font-size: 25px;
#                 position: relative;
#                 top: 80px;
#             }
#         </style>
#         """,
#         unsafe_allow_html=True,
#     )

# add_side_title()

############################# start page content #############################

st.title("References")
st.divider()

st.markdown("""
1. Brownlee, J. (2021, March 16). Smote for imbalanced classification with python. MachineLearningMastery.com. https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/ 
            
2. Centers for Disease Control and Prevention. (n.d.-a). About covid-19. Centers for Disease Control and Prevention. https://www.cdc.gov/coronavirus/2019-ncov/your-health/about-covid-19.html#:~:text=COVID%2D19%20(coronavirus%20disease%202019,%2C%20the%20flu%2C%20or%20pneumonia. 
            
3. Centers for Disease Control and Prevention. (n.d.-b). Covid-19 case surveillance public use data with geography. Centers for Disease Control and Prevention. https://data.cdc.gov/Case-Surveillance/COVID-19-Case-Surveillance-Public-Use-Data-with-Ge/n8mc-b4w4/about_data 
            
4. Coronavirus cases:. Worldometer. (n.d.). https://www.worldometers.info/coronavirus/ 
            
5. Randomundersampler#. RandomUnderSampler - Version 0.12.0. (n.d.). https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.RandomUnderSampler.html 
            
6. United States population by age - 2023 united states age demographics. Neilsberg. (n.d.). https://www.neilsberg.com/insights/united-states-population-by-age/#pop-by-age 
""")