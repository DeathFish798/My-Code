import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error as MSE



class svm_regression():
    '''
        svm支持向量机
        X:特征样本
        y:特征标签
        C:惩罚参数，数据越高泛化能力越弱
        kernal:核函数,线性'linear',非线性'rbf'
    '''
    def __init__(self,X,y,C = 2,kernel = 'rbf'):
        self.X = X
        self.y = y
        self.C = C
        self.k = kernel

    def __2Ddraw__(self):
        plt.scatter(self.X,self.y)
        plt.show()

    def __parameter__(self, lb=0.1, ub=2, num=19):
        '''选择参数'''
        # 划分训练集
        l1_tr = []
        l1_te = []
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(self.X, self.y, test_size=0.3, random_state=20)
        # 表现
        for i in np.linspace(lb,ub,num):
            l1_lr = SVR(C=i, kernel=self.k).fit(Xtrain, Ytrain)
            l1_tr.append(MSE(l1_lr.predict(Xtrain), Ytrain))
            l1_te.append(MSE(l1_lr.predict(Xtest), Ytest))
        graph = [l1_tr, l1_te]
        color = ['green', 'lightgreen']
        label = ['train', 'test']
        plt.figure()
        for i in range(len(graph)):
            plt.plot(np.linspace(lb,ub,num), graph[i], color[i], label=label[i])
        plt.legend(loc=4)
        plt.title('C')
        plt.show()

    def __SVR__(self,test_size = 0.3):
        '''
            SVR回归
            test_size: 测试集占比
        '''
        #回归
        lr = SVR(C=self.C,kernel=self.k).fit(self.X, self.y)
        self.lr = lr
        # 可能的画图
        NX = np.array(self.X)
        if NX.shape[1] <= 1:
            xx = np.linspace(int(min(self.X)) - 1, int(max(self.X)) + 1)
            xx = xx.reshape(-1,1)
            y_pred = self.lr.predict(xx)
            plt.scatter(self.X, self.y)
            plt.plot(xx, y_pred, c='r')
            plt.show()

    def __predict__(self,X):
        '''结果预测'''
        X = np.array(X)
        Y = self.lr.predict(X)
        print("预测值y:",Y)
        return Y

    def __evaluate__(self):
        '''结果评价'''
        y_pred = self.lr.predict(self.X)
        mse = MSE(y_true=self.y, y_pred=y_pred)
        print('MSE:', mse)


if __name__ == "__main__":
    #载入数据
    np.random.seed(0)
    x = np.random.uniform(-3, 3, size=100)  # 生成-3到3的100个随机数
    X = x.reshape(-1, 1)  # 变成二维数组
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)  # 定义y=0.5x^2+x+2的函数，加上噪音

    reg = svm_regression(X,y)
    reg.__2Ddraw__()
    reg.__parameter__(1,20,19)
    reg.C = 5
    reg.__SVR__()
    reg.__evaluate__()