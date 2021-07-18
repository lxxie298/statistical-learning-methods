import numpy as np
import pandas as pd
import time

def load_data(path):
    df=pd.read_csv(path,header=None)
    data=df.drop(0,1).values
    label=df[0].values
    return data,label

def calDis(x,y):
    return np.sqrt(np.sum(np.square(x-y)))

def getClosest(data,labels,x,topK):
    m,n=data.shape
    dis=[0]*m
    for i in range(m):
        y=data[i]
        dis[i]=calDis(x,y)
    indexList=np.argsort(dis)[:topK]

    cntClass=[0]*10
    for i in range(topK):
        cntClass[labels[indexList[i]]]+=1
    index = cntClass.index(max(cntClass))
    return index

def model_test(train_data,train_labels,test_data,test_labels,topK):
    m,n=test_data.shape
    cnt_err=0
    for i in range(m):
        print("current iterator:{0}:{1}".format(i+1,m))
        predicted=getClosest(train_data,train_labels,test_data[i],topK)
        if(predicted != test_labels[i]):
            cnt_err+=1
    print("accurate:{0}".format(1-cnt_err/m))

print("Loading..")
train_data,train_labels=load_data("../mnist_train.csv")
test_data,test_labels=load_data("../mnist_test.csv")
test_data=test_data[:200]
test_labels=test_labels[:200]

print("begin!")
start=time.time()
model_test(train_data,train_labels,test_data,test_labels,25)
end=time.time()
print("spent time:",end-start)
        
