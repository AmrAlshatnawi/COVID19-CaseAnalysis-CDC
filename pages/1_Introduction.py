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
st.divider()

st.markdown("""
The COVID-19 pandemic emerged as a catastrophic global event, touching every corner of the world.
This unprecedented crisis led to significant changes in our daily routines, leaving an indelible impact on our lives and
the way we work, even after its conclusion. To date, nearly **704 million** individuals have been impacted by COVID-19,
with the death toll surpassing **7 million**. Notably, the United States accounts for over a million of these fatalities,
as reported by the CDC. Through this analysis, our objective is to delve into the United States' COVID-19 data to uncover
insights regarding the pandemic's effects across various regions and demographic groups. We aim to determine if certain age
groups were more susceptible to infection and also explore whether specific variables can predict mortality rates.
""")

#st.image("./covid.png")



import base64

def get_image_base64(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Path to image
image_path = "./covid.png"

# Convert your image file to a base64 string
image_base64 = get_image_base64(image_path)
image_html = f'<img src="data:image/png;base64,{image_base64}" class="custom-img">'

# Custom CSS to position the image
css = """
<style>
.custom-img {
    position: fixed;
    bottom: 20px;
    right: 130px;
    width: 550px;  # Adjust the size as needed
}
</style>
"""

# Inject custom CSS and the base64-encoded image
st.markdown(css, unsafe_allow_html=True)
st.markdown(image_html, unsafe_allow_html=True)

