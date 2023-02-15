import streamlit as st
import pickle
import sys
sys.path.append("src/")
from src.pre_process import feature_process

from src.predict import classify
model = pickle.load(open("./models/lrc.pkl", "rb"))
       

def inputs():

    st.title("Bank loan prediction")

    gen=st.selectbox('Gender',['Male','Female'])
    mar=st.selectbox('Married',('Yes','No'))
    dep=st.selectbox('Dependents',('0','1','2','3','3+'))
    edu=st.selectbox('Education',('Not graduate','Graduate')) 
    emp=st.selectbox('Self employed',('Yes','No'))
    prop=st.selectbox('Property area',('Rural','Urban','Semiurban'))
    appincome=st.number_input('Applicants income',value=0)
    coappincome=st.number_input('Coapplicants income',value=0)
    loanamount=st.number_input('Loan amount',value=0)
    term=st.selectbox('Loan term',(12,36,60,84,120,180,240,300,360,480))
    crehist = st.selectbox("Credit history", (0,1))

    if st.button("Submit"):
        total=appincome+coappincome
        data=feature_process(gen,mar,dep,edu,emp,total,loanamount,term,crehist,prop)
        prediction=classify(data,model)
        st.write("Prediction for your application is {} ".format(prediction))

inputs()