#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 

# data processing
import pandas as pd 

# data visualization
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib import style
import streamlit as st
import pycaret


# In[3]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sub = pd.read_csv('gender_submission.csv')


# In[4]:


train.head()


# In[7]:


train["Survived"]=train["Survived"].apply(lambda x:"Survived" if x==1 else "Dead")


# In[8]:


test.head()


# In[9]:


train


# In[28]:


train["Embarked"].unique()


# In[5]:


train.info()


# In[6]:


def cat_summary(dataframe, col_name, plot=False):

    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


# In[7]:


cat_summary(train, "Survived", plot=True)


# In[8]:


from pycaret.classification import *


# In[9]:


clf1 = setup(data = train, 
             target = 'Survived',
             numeric_imputation = 'mean',
             categorical_features = ['Sex','Embarked'], 
             ignore_features = ['Name','Ticket','Cabin'],
             silent = True,
             log_experiment = True, 
             experiment_name = 'titanic'
             )


# In[10]:


compare_models()


# In[11]:


g_boost  = create_model('gbc') 


# In[12]:


tuned_gb = tune_model(g_boost)


# In[ ]:


best = compare_models(n_select = 15)
compare_model_results = pull()


# In[13]:


plot_model(estimator = tuned_gb, plot = 'learning')


# In[14]:


plot_model(estimator = tuned_gb, plot = 'auc')


# In[15]:


evaluate_model(tuned_gb)


# In[16]:


predictions=predict_model(tuned_gb,train)
predictions


# In[17]:


test_predictions=predict_model(tuned_gb,test)
test_predictions


# In[29]:


save_model(g_boost , 'deployment_28042021')


# In[31]:


deployment_28042021 = load_model('deployment_28042021')


# In[20]:


deployment_28042020


# In[ ]:





# In[ ]:


def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    model_index = list(compare_model_results['Model']).index(model)
    model = best[model_index]
    pred = predict_model(model, df, raw_score=True)
    predictions = predictions_df['Label'][0]
    return {'Dead': pred['Score_Not_Survived'][0].astype('float64'), 
            'Survived': pred['Score_Survived'][0].astype('float64'),
            }
   # return predictions


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
        model = st.multiselect(list(compare_model_results['Model']), label="Models")
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


# In[ ]:


run()

