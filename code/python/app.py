import sys
import os
import streamlit as st

# Add the parent directory ('code/') to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Now you can import from `web_app/`
from web_app.predict_page import show_predict_page
# from web_app.explore_page import show_explore_page  # If using explore_page

# from explore_page import show_explore_page  # Ensure this is present

# Sidebar navigation
page = st.sidebar.selectbox("Choose a Page", ("Heart Disease Prediction", "Explore Data"))

if page == "Heart Disease Prediction":
    show_predict_page()
# else:
#     show_explore_page()  # Ensure this function exists in explore_page.py
