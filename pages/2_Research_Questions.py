import streamlit as st

st.set_page_config(
    page_title="Covid Data Analysis - Research Questions",
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


st.title("Research Questions")
st.divider()

st.subheader("1.Is there a significant difference in COVID-19 case counts between different age groups?")
st.markdown("""- **H0**: The distribution of COVID-19 cases across age groups is proportional to the population distribution of those age groups, indicating that age, relative to its population size, does not influence the likelihood of contracting COVID-19.
- **H1**: The distribution of COVID-19 cases across age groups is not proportional to the population distribution of those age groups, suggesting that, relative to their population size, certain age groups are more likely to contract COVID-19 than others.
""")

st.subheader("2. Do gender, age group, and case year significantly associate with COVID-19 mortality outcomes?")
st.markdown("""- **H0**: Gender, age group, and case year do not significantly predict COVID-19 mortality.
- **H1**: Gender, age group, and case year significantly predict COVID-19 mortality.""")



