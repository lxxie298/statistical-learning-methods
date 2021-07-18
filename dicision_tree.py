import numpy as np
import pandas as pd


data = {"age":[0,0,0,0,0,1,1,1,1,1,2,2,2,2,2],
        "have_job":[0,0,1,1,0,0,0,1,0,0,0,0,1,1,0],
        "have_home":[0,0,0,1,0,0,0,1,1,1,1,1,0,0,0],
        "credit":[0,1,1,0,0,0,1,1,2,2,2,1,1,2,0],
        "label":[0,0,1,1,0,0,0,1,1,1,1,1,1,1,0]
        }

K=2
num_class = 2
def calEnt(dataSet):
    n = dataSet.shape[0]
    num_Di = dataSet.iloc[:,-1].value_counts()
    p = num_Di / n
    return -np.sum(p*np.log2(p))


def calConditionalEnt(dataSet,A):
    n = dataSet.shape[0]
    features = dataSet.iloc[:,A].value_counts().index
    hda = 0
    for f in features:
        childSet = dataSet[dataSet.iloc[:,A]==f]
        hda+=(childSet.shape[0]/n)*calEnt(childSet)
    return hda

def calGain(dataSet):
    ent = calEnt(dataSet)
    num_feature = dataSet.shape[1]-1
    gains = []
    for A in range(num_feature):
        cent = calConditionalEnt(dataSet,A)
        gains.append(ent - cent)
    return gains

def createDataSet(data):
    dataSet = pd.DataFrame(data)
    return dataSet

def bestSpilt(dataSet,axis,value):
    col_name = dataSet.columns[axis]
    childSet = dataSet[dataSet.iloc[:,axis]==value].drop(col_name,axis=1)
    return childSet

def createTreeByID3(dataSet,threshold):
    feature_list = list(dataSet.columns)
    class_list = dataSet.iloc[:,-1].value_counts().index
    if(len(class_list)==1 or len(feature_list)==1):
        return class_list[0]
    gains = calGain(dataSet)
    if(np.max(gains)<threshold):
        return dataSet.iloc[:,-1].value_counts(ascending=True).index[0]
    best_axis = np.argmax(gains)
    col_name = feature_list[best_axis]
    Tree = {col_name:{}}
    val_list = dataSet.iloc[:,best_axis].value_counts().index
    for val in val_list:
        Tree[col_name][val] = createTreeByID3(bestSpilt(dataSet,best_axis,val),threshold)
    return Tree

        

def classification(Tree,labels,test):
    feature_str = next(iter(Tree))
    feature_idx = labels.index(feature_str)
    for key in Tree[feature_str].keys():
        if(key == test[feature_idx]):
            if(type(Tree[feature_str][key])==dict):
                pred = classification(Tree[feature_str][key],labels,test)
            else:
                pred = Tree[feature_str][key]
    return pred

def acc_clasify(Tree,testSet):
    labels = list(testSet.columns)
    pred_list = []
    for i in range(testSet.shape[0]):
        pred = classification(Tree,labels,testSet.iloc[i])
        pred_list.append(pred)
    pred_list = np.array(pred_list)
    acc = np.sum(pred_list==testSet.iloc[:,-1].values)/testSet.shape[0]
    return acc
        

dataSet = createDataSet(data)
train = dataSet.iloc[:9]
test = dataSet.iloc[9:]

print("train len:",len(train))
print("test len",len(test))
Tree = createTreeByID3(train,0)
acc = acc_clasify(Tree,test)
        
