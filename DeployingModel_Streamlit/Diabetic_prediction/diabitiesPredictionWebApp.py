import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('DeployingModel_Streamlit\Diabetic_prediction\trained_model.sav','rb'))

# creating a function to predict the diabetic status
def diabetic_prediction(input_data):
    # changing the input data into numpy array
    input_arr = np.asarray(input_data)
    reshape_arr = input_arr.reshape(1,-1)

    prediction = loaded_model.predict(reshape_arr)
    print(prediction)

    if prediction[0] == 0:
        return('The person is not diabetic')
    else:
        return('The person is diabetic')

def main():

    # giving the title
    st.title('Diabetic Status Prediction Web App')

    # getting the input data from the user
    st.write('Please enter the following details')
    Pregnancies = st.text_input('No. of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure Value')
    SkinThickness = st.text_input('Skin Thickness')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function')
    Age = st.text_input('Age of the person')

    # code for the prediction
    diagnosis = ''

    # creating a button for prediction
    if st.button('Diabetic Test result'):
        diagnosis = diabetic_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])

    st.success(diagnosis)

if __name__ == '__main__':
    main()