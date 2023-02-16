
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def cleaning(data):
#Filling missing values
    df=data
    df.Dependents.fillna(df.Dependents.mode()[0],inplace=True)
    df.Loan_Amount_Term.fillna(df.Loan_Amount_Term.mode()[0],inplace=True)
    df.Gender.fillna(df.Gender.mode()[0],inplace=True)
    df.Married.fillna(df.Married.mode()[0],inplace=True)
    df.Self_Employed.fillna(df.Self_Employed.mode()[0],inplace=True)
    df.LoanAmount.fillna(df.LoanAmount.median(),inplace=True)
    df.loc[ (pd.isnull(df['Credit_History'])) & (df['Loan_Status'] == 'Y'), 'Credit_History'] = 1
    df.loc[ (pd.isnull(df['Credit_History'])) & (df['Loan_Status'] == 'N'), 'Credit_History'] = 0

#Encoding categories to numeric values
    df['gender']=LabelEncoder().fit_transform(df.Gender)
    df['maritalstatus']=LabelEncoder().fit_transform(df.Married)
    df['education']=LabelEncoder().fit_transform(df.Education)
    df['selfemployed']=LabelEncoder().fit_transform(df.Self_Employed)
    df['area']=LabelEncoder().fit_transform(df.Property_Area)
    df['loanstatus']=LabelEncoder().fit_transform(df.Loan_Status)
    df["Dependents"].replace({"3+": "3"}, inplace=True)

#Logarithmic values for higher numbers
    df['log_income']=np.log(df['ApplicantIncome']+df['CoapplicantIncome'])
    df['log_loanamt']=np.log(df.LoanAmount)
    df['log_loanterm']=np.log(df.Loan_Amount_Term)

#Dropping unwanted columns
    data=df.drop(['Gender','Married','Education','Property_Area','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Status','Loan_Amount_Term'],axis=1)
    return data
