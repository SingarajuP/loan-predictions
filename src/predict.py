import pandas as pd
import numpy as np

def feature_process(gen,mar,dep,edu,emp,total,loanamount,term,crehist,prop):
    test=pd.DataFrame()
   # test=pd.DataFrame({'gender':[data[0]],'maritalstatus':[data[1]],'Dependents':[data[2]],
    #                   'education':[data[3]],'selfemployed':[data[4]],'log_income':[data[5]],
   #                    'log_loanamt':[data[6]],'log_loanterm':[data[7]],'Credit_History':[data[8]],
   #                    'area':[data[9]]})
    test=pd.DataFrame({'gender':[gen],'maritalstatus':[mar],'Dependents':[dep],
                       'education':[edu],'selfemployed':[emp],'log_income':[total],
                       'log_loanamt':[loanamount],'log_loanterm':[term],'Credit_History':[crehist],
                       'area':[prop]})
    test.gender=test.gender.map({'Male':1,'Female':0})
    test.maritalstatus=test.maritalstatus.map({'Yes':1,'No':0})
    test.Dependents=test.Dependents.map({'0':0,'1':1,'2':2,'3':3,'3+':3})
    test.education=test.education.map({'Graduate':0,'Not graduate':1})
    test.selfemployed=test.selfemployed.map({'Yes':1,'No':0})
    test.log_income=np.log(test.log_income)
    test['log_loanamt']=np.log(test['log_loanamt'])
    test['log_loanterm']=np.log(test['log_loanterm'])
    test.area=test.area.map({'Rural':0,'Semiurban':1,'Urban':2})
    return test

def classify(data,model):
    label_decoder={0:'Loan will not be approved',1: 'Loan will be approved'}
    pred=model.predict(data)
    predi=label_decoder.get(pred[0])
    return predi
    