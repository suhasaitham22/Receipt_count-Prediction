import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
import itertools
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Load data
df = pd.read_csv('data_daily.csv')

# Sidebar navigation
page = st.sidebar.selectbox("Go to", ("Home", "Feature Engineering, Preprocessing, EDA", "Model Predictions"))

# Home page
if page == "Home":
    st.title("Fetch Rewards ML Engineer Exercise - Self-Assessment & Application")
    st.write("Welcome! This is my approach to Fetch Rewards' Machine Learning Engineer position.")
    
    st.subheader("Task - Self-Assessment")
    st.write("""
    As a candidate applying for the ML Engineer position, my assessment and approach are as follows:
    
    **Instructions & Approach:**
    - Understanding the task: Developed an ML model predicting scanned receipt numbers for 2022 based on 2021 data.
    - Model Development: Aimed for a robust yet interpretable algorithm suitable for future data.
    
    **Evaluation Metrics Used:**
    - Considering metrics aligned with prediction accuracy, such as RMSE or MAE and R squared for Regression Models.
    
    **Why Apply:**
    - Passionate about leveraging data-driven insights for impactful solutions.
    - Desire to contribute to Fetch Rewards' growth and innovation.
    """)

    st.subheader("Why Fetch Rewards")
    st.write("""
    **Reasons for Applying:**
    - Fetch's focus on innovation and utilizing data aligns with my career aspirations.
    - Excited about contributing to Fetch's success through innovative ML solutions.
    """)

    st.subheader("Why Me - Suhas Aitham")
    st.write("""
    **Strengths & Experience:**
    - Master's in Data Science & Business Analytics: Equipped with the necessary technical skills.
    - Proficient in various ML frameworks and technologies critical for this role.
    - Prior experience in predictive modeling, ETL pipelines, and developing interactive dashboards.
    """)
    
    st.subheader("Get Started - Fetch Rewards ML Engineer Exercise:")
    st.write("Navigate to 'Feature Engineering, Preprocessing, EDA' to explore the processed data and 'Model Predictions' to start predicting scanned receipts.")



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
    show_top_days = st.checkbox("Show Top 5 days with highest Receipts Count for selected month")

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
    
    # Only display if the checkbox is checked
    if show_top_days:
        selected_month = st.selectbox("Select a month (1-12)", options=range(1, 13), index=7)
        if selected_month:
            # Function to get top days for a specific month
            def top_days_for_month(month, top_n=5):
                selected_month_data = df[df['Month'] == month]
                top_days = selected_month_data.nlargest(top_n, 'Receipt_Count')
                return top_days.drop(columns='Year')
    
            # Get top 5 days for the selected month
            top_days = top_days_for_month(selected_month)
            st.subheader(f"Top 5 days with highest Receipts Count for month {selected_month}")
            st.write(top_days)

# Model Predictions page
elif page == "Model Predictions":
    st.title("Model Predictions")
    st.write("Here you can make predictions using your model.")

    # Selecting a model using a dropdown list
    selected_model = st.selectbox("Select Model", ("ARIMA", "Linear Regression", "Random Forest", "XGBoost", "PyTorch Model"))

    if selected_model == "ARIMA":
        st.write("ARIMA Model selected.")
        
        df.set_index('# Date', inplace=True)
        # Selecting only the 'Receipt_Count' column for time series forecasting
        ts_data = df['Receipt_Count']
    
        # Train-test split
        train_size = int(len(ts_data) * 0.8)
        train = ts_data.iloc[:train_size]
        test = ts_data.iloc[train_size:]
    
        # Grid search for optimal ARIMA parameters
        p = d = q = range(0, 3)
        pdq = list(itertools.product(p, d, q))
    
        best_rmse = np.inf
        best_params = None
    
        for param in pdq:
            try:
                model = ARIMA(train, order=param)
                arima_model = model.fit()
                predictions = arima_model.forecast(steps=len(test))
                rmse = np.sqrt(np.mean((predictions - test.values) ** 2))
    
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = param
            except:
                continue
    
        st.write(f"Best parameters: {best_params} with RMSE: {best_rmse}")
    
        # Fit ARIMA model with best parameters
        model = ARIMA(train, order=best_params)
        arima_model = model.fit()
    
        # Forecast
        predictions = arima_model.forecast(steps=len(test))
    
        # Display plot based on user's choice
        show_plot = st.checkbox("Display Plot")
        if show_plot:
            # Plotting
            plt.figure(figsize=(10, 6))
            plt.plot(test.index.to_numpy(), test.values, label='Actual')
            plt.plot(test.index.to_numpy(), predictions.to_numpy(), label='ARIMA Forecast', color='red')
            plt.xlabel('Date')
            plt.ylabel('Receipt Count')
            plt.title('ARIMA Forecast vs Actual')
            plt.legend()
            st.pyplot(plt)

    
        # Display predictions DataFrame based on user's choice
        show_predictions_df = st.checkbox("Display Predictions DataFrame")
        if show_predictions_df:
            # Create DataFrame with Date, Actual, and Predicted values
            predictions_df = pd.DataFrame({
                'Date': test.index,
                'Actual': test.values,
                'Predicted': predictions
            })
            st.write("Predictions DataFrame")
            st.write(predictions_df.reset_index(drop=True))

    
    elif selected_model == "Linear Regression":
        st.write("Linear Regression Model selected.")
        # Your Linear Regression model code here...

    elif selected_model == "Random Forest":
        st.write("Random Forest Model selected.")
        # Your Random Forest model code here...

    elif selected_model == "XGBoost":
        st.write("XGBoost Model selected.")
        # Your XGBoost model code here...

    elif selected_model == "PyTorch Model":
        st.write("PyTorch Model selected.")
        # Your PyTorch model code here...
