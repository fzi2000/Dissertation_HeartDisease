import streamlit as st
from predict_page import show_predict_page
# from explore_page import show_explore_page  # Ensure this is present

# Sidebar navigation
page = st.sidebar.selectbox("Choose a Page", ("Heart Disease Prediction", "Explore Data"))

# if page == "Heart Disease Prediction":
show_predict_page()
# else:
#     show_explore_page()  # Ensure this function exists in explore_page.py
