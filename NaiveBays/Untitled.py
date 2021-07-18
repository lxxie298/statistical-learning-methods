import pandas as pd
import numpy as np
import time
num_class=10
num_features=784

def loadData(path):
    df=pd.read_csv(path,header=None)
    data=df.drop(0,axis=1).values
    labels=df[0].values
    return data,labels

def getAllProbability(data,labels):
    m,n=data.shape
    cnt=[0]*num_class
    for i in range(m):
        cnt[labels[i]]+=1

    Py=[0]*num_class
    for i in range(num_class):
        Py[i] = np.log((cnt[i]+1)/(m+10))

    Py_x=np.zeros((num_class,n,256))
    for i in range(m):
        label=labels[i]
        x=data[i]
        for j in range(n):
            Py_x[label][j][x[j]]+=1

    for i in range(num_class):
        for j in range(n):
            for a in range(256):
                Py_x[i][j][a]=np.log((Py_x[i][j][a]+1)/(cnt[i]+256))

    return Py,Py_x

def naiveBays(Py,Py_x,x):
    P=[0]*num_class
    for i in range(num_class):
        s=0
        for j in range(num_features):
            s+=Py_x[i][j][x[j]]
        s+=Py[i]
        P[i]=s
    return P.index(max(P))
    

def test_model(Py,Py_x,test_data,test_labels):
    num_correct=0
    num_err=0
    m,n=test_data.shape
    for i in range(m):
        predicted = naiveBays(Py,Py_x,test_data[i])
        if(predicted!=test_labels[i]):
            num_err+=1
    return 1-(num_err/m)
            

print("Loading ...")
train_data,train_labels=loadData("../mnist_train.csv")
test_data,test_labels=loadData("../mnist_test.csv")
start = time.time()
print("Training ...")
Py,Py_x=getAllProbability(train_data,train_labels)
end = time.time()

print("Predicting ...")
accurate = test_model(Py,Py_x,test_data,test_labels)
print("Accurate:{0}% \nCost time:{1}".format(accurate*100,end-start))

