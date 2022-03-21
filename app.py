import pandas as pd
import time
import streamlit as st
import plotly.express as px


from pycaret.classification import *
import streamlit as st
import pandas as pd
import numpy as np

model = load_model('deployment_28042021')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions


def run():

    from PIL import Image
    
    #image_hospital = Image.open('titanic.png')

    #st.image(image,use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    #st.sidebar.info('This app is created to predict patient hospital charges')
    #st.sidebar.success('https://www.pycaret.org')
    
    #st.sidebar.image(image_hospital)

    st.title("Titanic Prediction App")

    if add_selectbox == 'Online':

        Age = st.number_input('Age', min_value=1, max_value=100, value=25)
        Sex = st.selectbox('Sex', ['male', 'female'])
        Pclass= st.number_input('P Class', 1,3)
        SibSp=  st.multiselect('Number of Siblings And Spouse',[0,1,2,3,4,5,8])
        Parch= st.multiselect('Parch',[0,1,2,3,4,5,6])
        Fare=  st.slider('Fare', 0,600)
        Embarked = st.selectbox('Embarked', ['S', 'C', 'Q'])
    
        output=""

        input_dict = {'Age' : Age, 'Sex' : Sex, 'Pclass':Pclass,'SibSp':SibSp,'Parch':Parch,'Fare':Fare,'Embarked':Embarked}
        input_df = pd.DataFrame([input_dict])
        st.dataframe(input_df) 
        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = '$' + str(output)

        st.success('The output is {}'.format(output))

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)


if __name__ == '__main__':
    run()

