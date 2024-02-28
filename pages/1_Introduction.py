import streamlit as st

st.set_page_config(
    page_title="Covid Data Analysis - Introduction",
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

st.title("Introduction")


