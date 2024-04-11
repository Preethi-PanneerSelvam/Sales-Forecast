import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer
import streamlit as st
import re
st.set_page_config(layout="wide")

st.write("""
<div style='text-align:center'>
    <h1 style='color:#009999;'>Sales Forecasting Application</h1>
</div>
""", unsafe_allow_html=True)

selected_tab = st.sidebar.radio("Select Model", ("XGBoostRegressor",))

if selected_tab == "XGBoostRegressor":
    # Define the possible values for the dropdown menus
    store_values = list(range(1, 46))
    department_values = list(range(1, 100))
    Type = ["A", "B", "C"]
    IsHoliday = [True, False]
    Day_Date = list(range(1, 32))
    Month_Date = list(range(1, 13))

    # Define the widgets for user input
    with st.form("my_form"):
        col1, col2, col3 = st.columns([5, 2, 5])
        with col1:
            st.write(' ')
            store = st.selectbox("store", store_values, key=1)
            department = st.selectbox("Department", department_values, key=2)
            Temperature = st.text_input("Enter Temperature in Celsius (Min : -2.0, Max : 46.5)")
            Fuel_Price = st.text_input("Enter Fuel Price (Min : 2.4, Max : 4.5)")
            MarkDown1 = st.text_input("Enter MarkDown1")
            MarkDown2 = st.text_input("Enter MarkDown2")
            MarkDown3 = st.text_input("Enter MarkDown3")
            MarkDown4 = st.text_input("Enter MarkDown4")
            MarkDown5 = st.text_input("Enter MarkDown5")
            Type = st.selectbox("Select the Type of Store", Type, key=3)
            IsHoliday = st.selectbox("Holiday", IsHoliday, key=4)

            # Convert the selected values to binary
            IsHoliday = 1 if IsHoliday else 0
            Type_B = 1 if Type == "B" else 0
            Type_C = 1 if Type == "C" else 0

        with col3:
            st.write('<h5 style="color:rgb(0, 153, 153,0.4);">NOTE: Min & Max given for reference, you can enter any value</h5>', unsafe_allow_html=True)
            CPI = st.text_input("Enter Consumer Price Index (Min : 126, Max : 212)")
            Unemployment = st.text_input("Enter Unemployment percentage(Min : 3.8, Max : 14.4)")
            Size = st.text_input("Enter the size of the store (Min : 34875, Max : 219622)")
            Day_Date = st.selectbox("Day", Day_Date, key=5)
            Month_Date = st.selectbox("Month", Month_Date, key=6)
            year_date = st.text_input("Enter the year")
            submit_button = st.form_submit_button(label="PREDICT SELLING PRICE")
            st.markdown("""
                <style>
                div.stButton > button:first-child {
                    background-color: #009999;
                    color: white;
                    width: 100%;
                }
                </style>
            """, unsafe_allow_html=True)

        flag = 0
        pattern = "^(?:\d+|\d*\.\d+)$"
        for i in [Fuel_Price, MarkDown1, MarkDown2, MarkDown3, MarkDown4, MarkDown5, CPI, Unemployment, Size, year_date, Temperature]:
            if re.match(pattern, i):
                pass
            else:
                flag = 1
                break

    if submit_button and flag == 1:
        if len(i) == 0:
            st.write("please enter a valid number space not allowed")
        else:
            st.write("You have entered an invalid value: ", i)

    if submit_button and flag == 0:

        import pickle 
        with open(r"D:\SALESPROJECT\model.pkl", 'rb') as file:
            loaded_model = pickle.load(file)
        with open(r"D:\SALESPROJECT\scaler.pkl", 'rb') as f:
            scaler_loaded = pickle.load(f)

        scale_columns = np.array([[float(Temperature), float(Fuel_Price), float(MarkDown1), float(MarkDown2), float(MarkDown3), float(MarkDown4), float(MarkDown5), float(CPI), float(Unemployment), float(Size)]])
        
        # Handle zero values by adding a small epsilon
        epsilon = 1e-8  # Small value to avoid taking logarithm of zero
        scale_columns[scale_columns == 0] = epsilon

        scale_columns = np.log(scale_columns)  # Take logarithm of the features
        new_sample = scaler_loaded.transform(scale_columns)
        f = np.array([[int(store), int(department)]])
        l = np.array([[int(Month_Date), int(Day_Date), int(year_date), int(Type_B), int(Type_C), int(IsHoliday)]])
        new_sample1 = np.concatenate((f, new_sample, l), axis=1)

        new_pred = loaded_model.predict(new_sample1)[0]
        st.write('## :green[Predicted Weekly Sales:] ', new_pred)
