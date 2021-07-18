import numpy as np
import pandas as pd
import time

def perceptron(data,label,it=50):
    data=np.mat(data)
    label=np.mat(label).T
    m,n=data.shape
    w=np.zeros((1,n))
    b=0
    h=0.0001
    for k in range(it):
        err=0
        for i in range(m):
            x=data[i]
            y=label[i]
            if (y*(w*x.T+b))<=0:
                w=w+h*y*x
                b=b+h*y
                err+=1
        print("Round {0}:{1} training,acc={2}".format(k,it,1-err/m))
    return w,b
                

def model_test(data,label,w,b):
    m,n=data.shape
    data=np.mat(data)
    label=np.mat(label).T
    num_acc=0
    for k in range(m):
        x,y=data[k],label[k]
        if (y*(w*x.T+b))>0:
            num_acc+=1
    return num_acc/m
    

def loadData(path):
    df=pd.read_csv(path,header=None)
    data=df.drop(0,axis=1)
    data=np.array(data,dtype='float')
    label=df[0]
    label=np.array(label)
    for i in range(len(label)):
        if label[i]>=5:
            label[i]=1
        else:
            label[i]=-1
    data/=255
    return data,label

start=time.time()
train_data,train_label=loadData("../mnist_train.csv")
test_data,test_label=loadData("../mnist_test.csv")
w,b=perceptron(train_data,train_label,it=30)
end=time.time()
acc=model_test(test_data,test_label,w,b)
print("accuracy rate is:",acc)
print("time span:",end-start)

    

