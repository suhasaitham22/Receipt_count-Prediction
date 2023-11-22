import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data_daily.csv')

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

    # Checkbox options for displaying sections
    show_dataset = st.checkbox("Show Dataset")
    show_dataframe_info = st.checkbox("Show DataFrame Info")
    show_null_counts = st.checkbox("Show Null Value Counts")
    show_histogram = st.checkbox("Show Histogram")
    show_feature_engineered_dataframe = st.checkbox("Show Feature Engineered DataFrame")
    show_monthly_receipts_line_chart = st.checkbox("Show Monthly Receipts - Line Chart")
    show_monthly_receipts_bar_chart = st.checkbox("Show Monthly Receipts - Bar Chart (Matplotlib)")

    # Display sections based on user selection
    if show_dataset:
        st.subheader("Dataset")
        st.write(df)

    if show_dataframe_info:
        st.subheader("DataFrame Info")
        st.write(df.info())

    if show_null_counts:
        st.subheader("Null Value Counts")
        st.write(df.isnull().sum())

    if show_histogram:
        st.subheader("Histogram")
        column_to_plot = st.selectbox("Select column for histogram", df.columns)
        
        # Create a Matplotlib figure and plot the histogram
        fig, ax = plt.subplots()
        ax.hist(df[column_to_plot], bins=20)
        st.pyplot(fig)

    # Convert '# Date' column to datetime
    df['# Date'] = pd.to_datetime(df['# Date'])

    # Extracting year, month, and day into separate columns
    df['Year'] = df['# Date'].dt.year
    df['Month'] = df['# Date'].dt.month
    df['Day'] = df['# Date'].dt.day

    if show_feature_engineered_dataframe:
        st.subheader("Feature Engineered DataFrame")
        st.write(df)

    # Grouping by Month and summing 'Receipt_Count'
    monthly_receipts = df.groupby('Month')['Receipt_Count'].sum().reset_index()

    if show_monthly_receipts_line_chart:
        st.subheader("Monthly Receipts - Line Chart")
        fig = px.line(monthly_receipts, x='Month', y='Receipt_Count',
                      labels={'Receipt_Count': 'Monthly Receipts', 'Month': 'Month'},
                      title='Monthly Receipts')
        st.plotly_chart(fig)

    if show_monthly_receipts_bar_chart:
        st.subheader("Monthly Receipts - Bar Chart (Matplotlib)")
        monthly_receipts = monthly_receipts.sort_values('Receipt_Count', ascending=False)
        plt.figure(figsize=(8, 6))
        plt.bar(monthly_receipts['Month'], monthly_receipts['Receipt_Count'], color='skyblue')
        plt.xlabel('Month')
        plt.ylabel('Receipt Count')
        plt.title('Monthly Receipts')
        plt.gca().invert_xaxis()
        st.pyplot(plt)
    

    # Selecting a month using a dropdown list
    selected_month = st.selectbox("Select a month (1-12)", options=range(1, 13), index=7)
    if selected_month:
        # Function to get top days for a specific month
        def top_days_for_month(month, top_n=5):
            selected_month_data = df[df['Month'] == month]
            top_days = selected_month_data.nlargest(top_n, 'Receipt_Count')
            return top_days.drop(columns='Year')

        # Get top 5 days for the selected month
        top_days = top_days_for_month(selected_month)
        st.subheader(f"Top 5 days for month {selected_month}")
        st.write(top_days)

# Model Predictions page
elif page == "Model Predictions":
    st.title("Model Predictions")
    st.write("Here you can make predictions using your model.")
    # Add your model prediction content here
