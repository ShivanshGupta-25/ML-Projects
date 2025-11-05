import numpy as np
import pandas as pd
import pickle
import streamlit as st

# loading the saved model
loadedModel = pickle.load(open('DeployingModel_Streamlit\Car_Price_Prediction\CarPrice_trainedmodel.sav','rb'))

# Creating a function to predict the price of the car
def PredictCarPrice(input_data):
    # changing the input data into pandas dataframe
    input_df = pd.DataFrame([input_data], columns=['Year','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner'])

    # converting the categorical data into numerical data
    input_df.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2},
                      'Seller_Type':{'Dealer':0,'Individual':1},
                      'Transmission':{'Manual':0,'Automatic':1}}, inplace=True)
    
    prediction = loadedModel.predict(input_df)
    return(prediction)

def main():
    st.title("Car Price Prediction Web App")
    st.write("Please enter the following details")
    Year = st.text_input("Year")
    Present_Price = st.text_input("Present Price")
    Kms_Driven = st.text_input("Kms Driven")
    Fuel_Type = st.text_input("Fuel Type - (Petrol, Diesel, CNG)")
    Seller_Type = st.text_input("Seller Type - (Dealer, Individual)")
    Transmission = st.text_input("Transmission - (Manual, Automatic)")
    Owner = st.text_input("Owner")

    # Creating a button for prediction
    if st.button("Predict"):
        prediction = PredictCarPrice([Year,Present_Price,Kms_Driven,Fuel_Type,Seller_Type,Transmission,Owner])
        st.success(prediction)

if __name__ == '__main__':
    main()