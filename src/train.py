import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from pre_process import cleaning
from evaluate import classification_metrics
from evaluate import confusionmatrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sys
sys.path.append("src/")

def train():
    df = pd.read_csv("./data/raw/data.csv") 

    df_train=cleaning(df)
    X=df_train.filter(['gender','maritalstatus','Dependents','education','selfemployed','log_income','log_loanamt','log_loanterm','Credit_History','area'],axis=1)
    y=df.filter(['loanstatus'],axis=1)

    Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size = 0.3, random_state=5,stratify=y)

    lrc = LogisticRegression()
    lrc.fit(Xtrain,ytrain)

    pred=lrc.predict(Xtest)
    classification_metrics(pred, ytest,Xtrain,ytrain,lrc)
    confusionmatrix(pred,ytest)

if __name__ == '__main__':
  train()