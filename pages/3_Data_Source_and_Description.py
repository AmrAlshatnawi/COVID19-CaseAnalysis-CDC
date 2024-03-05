import streamlit as st

st.set_page_config(
    page_title="Covid Data Analysis - Data Source",
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


st.title("Data Source and Description")
st.divider()