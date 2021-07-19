# coding:utf8
import numpy as np
import  matplotlib.pyplot as plt

def loadDataSet(fileName):
    """
    Args:
        fileName 文件名
    Returns:
        data  特征矩阵
        label 类标签
    """
    data = []
    label = []
    with open(fileName) as f:
        lines = f.readlines()
        for line in lines:
            arr = line.strip().split("\t")
            data.append(arr[0:2])
            label.append(arr[2])
    return data,label


class SVM():
    def __init__(self,data,label,C=0.6,epsilon=0.001,kernal_options=["linear",0]):
        """
        结构化SVM模型
        Args:
            data 数据集
            label 标签
            C 松弛程度
            epsilon 满足KKT条件的误差范围
            kernal 核名字
        Returns:

        """
        self.data = np.mat(np.array(data,dtype="float"))
        self.label = np.mat(np.array(label,dtype="float")).T
        self.m = self.data.shape[0]
        self.n = self.data.shape[1]
        self.C = C
        self.epsilon = epsilon
        self.kernal_name = kernal_options[0]
        self.kernal_param = kernal_options[1]
        self.K = self.getKMat()
        self.alpha = np.mat(np.zeros((self.m,1)))
        self.b = 0
    def getKMat(self):
        K = np.mat(np.zeros((self.m,self.m)))
        for i in range(self.m):
            K[:,i] = self.calKernal(self.data[i,:])
        return K
    
    def calKernal(self,x):
        k_val = np.mat(np.zeros((self.m,1)))
        x = np.mat(x)
        if self.kernal_name=="linear":
            k_val = self.data*x.T
        elif self.kernal_name == 'rbf':
            sigma = self.kernal_param
            if sigma == 0:
                sigma = 1.0
            for i in range(self.m):
                diff = self.data[i, :] - x
                k_val[i] = np.exp(diff * diff.T / (-2.0 * sigma**2))
        return k_val

    def calPerdictKernal(self,x,y):
        if self.kernal_name=="linear":
            k_val = x*y.T
        elif self.kernal_name == 'rbf':
            sigma = self.kernal_param
            if sigma == 0:
                sigma = 1.0
                diff = x - y
                k_val = np.exp(diff * diff.T / (-2.0 * sigma**2))
        return k_val
        
        

    def predict(self,x):
        """
        预测新数据x的标签
        Args:
            x 新数据
        Returns:
            预测标签
        """
        x = np.mat(np.array(x,dtype="float"))
        m = x.shape[0]
        predicted = np.zeros((m,1))
        for i in range(m):
            s = self.b
            for k in range(self.m):
                s += self.alpha[k]*self.label[k] * self.calPerdictKernal(self.data[k,:],x[i,:])
            if(s>=0):
                predicted[i] = 1 
            else:
                predicted[i] = -1
        return np.squeeze(predicted)
            
        
def selectJRand(i,m):
    """
    随机选择a2的下标 j!=i
    Args:
        i 第一个a的下标
        m 数据集数量
    Returns:
        随机下标j
    """
    j=i
    while(j == i):
        j = int(np.random.uniform(0,m))
    return j



def smoSimple(model, maxIter):
    """
    使用简单版本smo算法训练模型
    Args:
        model SVM模型
        maxIter 最大迭代次数
    Returns:
        model 训练好的模型
    """
    Iter = 0 #迭代次数
    while(Iter < maxIter):
        isChanged = False
        for i in range(model.m):
            fxi = float(np.multiply(model.alpha, model.label).T * model.K[:,i] + model.b) # 计算xi的预测值
            Ei = fxi - float(model.label[i]) # 计算误差
            #判断是否符合KKT条件
            if ((model.label[i]*Ei < -model.epsilon) and (model.alpha[i] < model.C)) or ((model.label[i]*Ei > model.epsilon) and (model.alpha[i] > 0)):
                j = selectJRand(i,model.m) # 随机选择aj
                fxj = float(np.multiply(model.alpha, model.label).T * model.K[:,j] + model.b) # 计算xj预测值
                Ej = fxj - float(model.label[j]) # 计算误差

                # 保存a1、a2的旧值
                ai_old = model.alpha[i].copy()
                aj_old = model.alpha[j].copy()

                eta = model.K[i,i] + model.K[j,j] - 2.0*model.K[i,j] # 计算η
                if(eta<=0):
                    continue
                
                if(model.label[i] != model.label[j]):
                    L = max(0,aj_old - ai_old)
                    H = min(model.C,model.C + aj_old - ai_old )
                else:
                    L = max(0,aj_old + ai_old - model.C)
                    H = min(model.C,aj_old + ai_old)
                if(L==H):
                    continue
                
                model.alpha[j] += model.label[j]*(Ei-Ej)/eta # 更新a[j]
                model.alpha[j] = np.clip(model.alpha[j],L,H) # 约束a[j]的范围
                # 更新范围太小，跳过a[i]
                if(abs(model.alpha[j] - aj_old) < 0.00001):
                    continue
                
                model.alpha[i] += model.label[i]*model.label[j]*(aj_old - model.alpha[j]) # 更新a[i]
                
                # 计算b1 b2
                b1 = -Ei - model.label[i]*model.K[i,i]*(model.alpha[i]-ai_old) - model.label[j]*model.K[i,j]*(model.alpha[j] - aj_old) + model.b
                b2 = -Ej - model.label[i]*model.K[i,j]*(model.alpha[i]-ai_old) - model.label[j]*model.K[j,j]*(model.alpha[j] - aj_old) + model.b

                # 更新b
                if ((0 < model.alpha[i]) and (model.alpha[i] < model.C)):
                    model.b = b1
                elif ((0 < model.alpha[j]) and (model.alpha[j] < model.C)):
                    model.b = b2
                else:
                    model.b = (b1+b2)/2.0
                              
                isChanged = True

        # 如果一轮下来，a向量没有被更新，则进入下一轮迭代，否则清0
        if(not isChanged):
            Iter += 1
        else:
            Iter = 0
    return model

def showSVM(svm):
    if svm.data.shape[1] != 2:
        return 0

    # 画数据集散点图
    for i in range(svm.m):
        if svm.label[i] == -1:
            plt.plot(svm.data[i, 0], svm.data[i, 1], 'or')
        elif svm.label[i] == 1:
            plt.plot(svm.data[i, 0], svm.data[i, 1], 'ob')

    # 标记支持向量
    supportVectorsIndex = np.nonzero(svm.alpha > 0)[0]
    
    for i in supportVectorsIndex:
        plt.plot(svm.data[i, 0], svm.data[i, 1], 'oy')
    

    # 画分类超平面

    
    w = np.zeros((2, 1))
    for i in supportVectorsIndex:
        w += np.multiply(svm.alpha[i] * svm.label[i], svm.data[i, :].T)
    min_x = np.min(svm.data[:, 0])
    max_x = np.max(svm.data[:, 0])
    y_min_x = float(-svm.b - w[0] * min_x) / w[1]
    y_max_x = float(-svm.b - w[0] * max_x) / w[1]
    plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
    
    plt.show()


if __name__ == "__main__":
    data, label = loadDataSet('testSet.txt')
    model = SVM(data,label)
    model = smoSimple(model,40)
    showSVM(model)
    
            
                
            
                
            
        
    
            
        
