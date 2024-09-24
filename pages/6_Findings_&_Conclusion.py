import streamlit as st

st.set_page_config(
    page_title="Covid Data Analysis - Findings & Conclusion",
    page_icon="📊",
    #layout="wide"   
)
#st.markdown("""<style>body {zoom: 1.4;  /* Adjust this value as needed */}</style>""", unsafe_allow_html=True)

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

st.title("Findings & Conclusion")
st.divider()



st.markdown("""

Our analysis aimed to uncover patterns in COVID-19 case counts and mortality outcomes, focusing on the roles of age, gender, and case year.
Initially, we explored the distribution of COVID-19 cases across various age groups. The data revealed a significant deviation in the 18 to 49 age group,
which displayed a disproportionately high number of cases. Utilizing the Chi-square goodness-of-fit test, we determined this variance to be statistically
significant, with a p-value less than 0.05. This finding suggests certain age groups, notably the 18 to 49 demographic, are more susceptible to contracting
COVID-19 relative to their population size, potentially due to factors like social behavior and employment types.

In examining COVID-19 mortality outcomes, logistic regression analysis highlighted that gender, age group, and case year are significant predictors of mortality,
with all predictors showing statistical significance (p-values < 0.05). Despite an initial dataset imbalance, our resampling strategy, which included both
undersampling and oversampling techniques, allowed us to maintain the model's overall significance while revealing an increased baseline probability of death
in a more balanced dataset context. This adjustment suggests a refined understanding of mortality risk factors. However, the model's precision at 10.62%
indicates a high rate of false positives, a challenge balanced by its strong sensitivity (81.44%) in accurately identifying actual deaths. This emphasizes
the model's utility in critical public health scenarios despite its need for further optimization to reduce false positives and improve the F1 score (0.1879).

In conclusion, our findings confirm the significant impact of gender, age, and case year on COVID-19 mortality, underscoring the importance of targeted public
health strategies. While the model presents areas for improvement, its ability to predict true positives remains a valuable asset in managing the pandemic
response, highlighting the potential for further refinement to enhance its predictive accuracy and applicability.


""")