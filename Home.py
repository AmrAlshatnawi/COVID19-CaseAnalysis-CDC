import streamlit as st

st.set_page_config(
    page_title="Covid Data Analysis",
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
st.title("Exploring the Facets of COVID-19: A Multi-Question Analysis")
st.write("✍️Authors: Amr Alshatnawi, Hailey Pangburn, Richard McMasters")
st.write("---")
st.header("MSBI 32000 Winter 2024 Semester - The University of Chicago")
st.write("""Welcome to our  COVID-19 data analysis web application, where we delve into the intricacies of the COVID-19 pandemic through data.
          Our mission is to uncover the hidden patterns, trends, and insights within the vast amounts of case surveillance data provided by the CDC.
          This project is a collaborative effort, aimed at enhancing our understanding of the pandemic's impact across different geographies and demographics.""")


st.markdown("""Here, you'll find a comprehensive analysis structured into various sections:
- **Introduction:** Get to know the background and objectives of our analysis.
- **Research Questions:** Explore the key questions guiding our investigation.
- **Data Source:** Learn about the CDC's COVID-19 Case Surveillance Public Use Data and how it's utilized in our study.
- **Data Exploration and Munging:** This section outlines our initial steps in preparing and understanding the dataset. We start by examining the structure, quality, and distribution of the COVID-19 Case Surveillance Public Use Data.
- **Analysis:** Dive into our findings with interactive visualizations and code snippets that bring the data to life.
- **Findings & Conclusion:** Discover the significant patterns and insights we've uncovered through our analysisand reflect on the implications of our findings and the potential paths forward.""")

st.info("""**Acknowledgement**  
This project was completed by students in the Intermediate Applied Data Analysis class within the Biomedical Informatics Program at the University of Chicago. Our work represents a collaborative educational endeavor, where we applied our learning to real-world data to contribute to the broader understanding of the COVID-19 pandemic.""")
