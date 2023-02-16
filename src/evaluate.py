import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score,recall_score, f1_score,accuracy_score, confusion_matrix

def classification_metrics(pred,ytest,Xtrain,ytrain,lrc):
    print("Accuracy of training data:",lrc.score(Xtrain,ytrain))
    print("Test data results:")
    print("Accuracy:",accuracy_score(ytest, pred))
    print("Precision:",precision_score(ytest, pred))
    print("Recall:",recall_score(ytest, pred))
    print("f1:",f1_score(ytest, pred))


def confusionmatrix(pred,ytest):
    cm = confusion_matrix(ytest, pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True,fmt='g')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.show()