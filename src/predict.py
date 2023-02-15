import pickle

def classify(data,model):
    label_decoder={0:'Loan will not be approved',1: 'Loan will be approved'}
    pred=model.predict(data)
    return pred,label_decoder.get(pred[0])
    