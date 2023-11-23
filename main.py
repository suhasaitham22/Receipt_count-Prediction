import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.api as sm
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
    df['day_number'] = df['# Date'].dt.dayofyear
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
    selected_model = st.selectbox("Select Model", ("ARIMA", "Linear Regression", "PyTorch Model"))
    
    if selected_model == "ARIMA":
        
        st.write("Here you can make predictions using your model.")
    
        # Convert index to datetime if it's not already
        df['# Date'] = pd.to_datetime(df['# Date'])
        
        # Resample data to monthly frequency
        ts_data = df.set_index('# Date')['Receipt_Count'].resample('M').sum()
        
        # Train-test split
        train_size = int(len(ts_data) * 0.8)
        train = ts_data.iloc[:train_size]
        test = ts_data.iloc[train_size:]
        
        # Using auto ARIMA to find the best parameters
        auto_arima_model = sm.tsa.arima.auto_arima(
            train,
            start_p=1, start_q=1,
            max_p=3, max_q=3,
            m=12,  # for monthly data
            start_P=0, seasonal=True,
            d=1, D=1,
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )
        
        # Fit ARIMA model with best parameters
        arima_model = sm.tsa.ARIMA(train, order=auto_arima_model.order)
        arima_fit = arima_model.fit()
        
        # Forecast
        monthly_predictions = arima_fit.forecast(steps=len(test))
        monthly_index = pd.date_range(start=test.index[0], periods=len(test), freq='M')
        
        # Display plot based on user's choice
        show_plot = st.checkbox("Display Plot")
        if show_plot:
            # Create a DataFrame for visualization
            plot_data = pd.DataFrame({
                'Month': monthly_index,
                'Actual': test.values,
                'ARIMA Forecast': monthly_predictions
            })
        
            # Plotting using Plotly Express
            fig = px.line(plot_data, x='Month', y=['Actual', 'ARIMA Forecast'], title='ARIMA Forecast vs Actual')
            fig.update_xaxes(title='Month')
            fig.update_yaxes(title='Receipt Count')
            st.plotly_chart(fig)
        
        # Display predictions DataFrame based on user's choice
        show_predictions_df = st.checkbox("Display Predictions DataFrame")
        if show_predictions_df:
            # Create DataFrame with Month, Actual, and Predicted values
            predictions_df = pd.DataFrame({
                'Month': monthly_index,
                'Actual': test.values,
                'Predicted': monthly_predictions
            })
            st.write("Predictions DataFrame")
            st.write(predictions_df.reset_index(drop=True))  

    
    elif selected_model == "Linear Regression":
        st.write('Linear Regression Model Selected')
        # Convert index to datetime if it's not already
        df['# Date'] = pd.to_datetime(df['# Date'])
        
        # Extract relevant features
        df['month'] = df['# Date'].dt.month
        df['quarter'] = df['# Date'].dt.quarter  # Add quarter as a feature
    
        # Lag features: Previous month's sales
        df['previous_month_sales'] = df['Receipt_Count'].shift(1)
        
        selected_columns = ['month', 'quarter', 'previous_month_sales', 'Receipt_Count']
        data = df[selected_columns].dropna()
    
        # Splitting data into features and target variable
        X = data.drop('Receipt_Count', axis=1)
        y = data['Receipt_Count']
    
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        # Creating a linear regression model
        model = LinearRegression()
    
        # Training the model
        model.fit(X_train, y_train)
    
        # Making predictions
        y_pred = model.predict(X_test)
    
        # Model evaluation
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
    
        st.write(f"Mean Squared Error: {mse}")
        st.write(f"R-squared Score: {r2}")
    
        # Display predictions DataFrame based on user's choice
        show_predictions_df = st.checkbox("Display Predictions DataFrame")
        if show_predictions_df:
            # Create DataFrame with features, Actual, and Predicted values
            predictions_df = pd.DataFrame({
                'Month': X_test['month'],
                'Quarter': X_test['quarter'],
                'Previous_Month_Sales': X_test['previous_month_sales'],
                'Actual': y_test,
                'Predicted': y_pred
            })
            st.write("Predictions DataFrame")
            st.write(predictions_df)

    elif selected_model == "PyTorch Model":
        st.write("PyTorch Model selected.")
        # Your PyTorch model code here...
