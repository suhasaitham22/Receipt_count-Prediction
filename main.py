import streamlit as st

# Sidebar navigation
page = st.sidebar.selectbox("Go to", ("Home", "Feature Engineering, Preprocessing, EDA", "Model Predictions"))

# Home page
if page == "Home":
    st.title("Home")
    st.write("Welcome to the Home page!")
    # Add any content for the Home page here

# Feature Engineering, Preprocessing, EDA page
elif page == "Feature Engineering, Preprocessing, EDA":
    st.title("Feature Engineering, Preprocessing, EDA")
    st.write("This is the Feature Engineering, Preprocessing, and EDA page.")
    # Add your EDA and preprocessing content here

# Model Predictions page
elif page == "Model Predictions":
    st.title("Model Predictions")
    st.write("Here you can make predictions using your model.")
    # Add your model prediction content here
