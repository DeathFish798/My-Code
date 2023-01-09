import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score


class linear_regression():
    '''
        线性回归
        X:特征样本
        y:特征标签
    '''
    def __init__(self,X,y):
        self.X = X
        self.y = y

    def __2Ddraw__(self):
        plt.scatter(self.X,self.y,marker='o')
        plt.show()

    def __LR__(self):
        '''
            线性回归
        '''
        #回归
        lr = LR().fit(self.X, self.y)
        self.lr = lr
        self.coef = lr.coef_
        self.intercept = lr.intercept_
        print()
        print("w:",self.coef)
        print("b:",self.intercept)
        print()
        #可能的画图
        NX = np.array(self.X)
        if NX.shape[1] <= 1:
            y_pred = self.lr.predict(self.X)
            plt.scatter(self.X,self.y)
            plt.plot(self.X,y_pred,c='r')
            plt.show()

    def __parameter__(self):
        '''选择参数'''
        #评价拟合效果
        l1_tr = []
        l1_te = []
        l2_tr = []
        l2_te = []
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(self.X, self.y, test_size=0.3, random_state=1)
        #MSE表现
        for i in np.linspace(0.05, 2, 38):
            l1_lr = Lasso(alpha=i).fit(Xtrain, Ytrain)
            l2_lr = Ridge(alpha=i).fit(Xtrain, Ytrain)
            # 比较L1正则化和L2正则化的模型效果差异
            l1_tr.append(MSE(l1_lr.predict(Xtrain), Ytrain))
            l1_te.append(MSE(l1_lr.predict(Xtest), Ytest))
            l2_tr.append(MSE(l2_lr.predict(Xtrain), Ytrain))
            l2_te.append(MSE(l2_lr.predict(Xtest), Ytest))
        graph = [l1_tr, l2_tr, l1_te, l2_te]
        color = ['green', 'black', 'lightgreen', 'gray']
        label = ['l1_train', 'l2_train', 'l1_test', 'l2_test']
        plt.figure()
        for i in range(len(graph)):
            plt.plot(np.linspace(0.05, 2, 38), graph[i], color[i], label=label[i])
        plt.legend(loc=2)
        plt.title('MSE')
        # R2表现
        l1_tr = []
        l1_te = []
        l2_tr = []
        l2_te = []
        for i in np.linspace(0.05, 2, 38):
            l1_lr = Lasso(alpha=i).fit(Xtrain, Ytrain)
            l2_lr = Ridge(alpha=i).fit(Xtrain, Ytrain)
            # 比较L1正则化和L2正则化的模型效果差异
            l1_tr.append(r2_score(l1_lr.predict(Xtrain), Ytrain))
            l1_te.append(r2_score(l1_lr.predict(Xtest), Ytest))
            l2_tr.append(r2_score(l2_lr.predict(Xtrain), Ytrain))
            l2_te.append(r2_score(l2_lr.predict(Xtest), Ytest))
        graph = [l1_tr, l2_tr, l1_te, l2_te]
        color = ['green', 'black', 'lightgreen', 'gray']
        label = ['l1_train', 'l2_train', 'l1_test', 'l2_test']
        plt.figure()
        for i in range(len(graph)):
            plt.plot(np.linspace(0.05, 2, 38), graph[i], color[i], label=label[i])
        plt.legend(loc=3)
        plt.title('R2')
        plt.show()

    def __Lasso__(self,alpha = 1):
        '''
            lasso回归
            L1正则化
            alpha:正则化系数
        '''
        lr = Lasso().fit(self.X,self.y)
        self.lr = lr
        self.coef = lr.coef_
        self.intercept = lr.intercept_
        print()
        print("w:", self.coef)
        print("b:", self.intercept)
        print()
        # 可能的画图
        NX = np.array(self.X)
        if NX.shape[1] <= 1:
            y_pred = self.lr.predict(self.X)
            plt.scatter(self.X, self.y)
            plt.plot(self.X, y_pred, c='r')
            plt.show()

    def __Ridge__(self,alpha = 1):
        '''
            Rigde回归
            L2正则化
            alpha:正则化系数
        '''
        lr = Ridge().fit(self.X,self.y)
        self.lr = lr
        self.coef = lr.coef_
        self.intercept = lr.intercept_
        print()
        print("w:", self.coef)
        print("b:", self.intercept)
        print()
        # 可能的画图
        NX = np.array(self.X)
        if NX.shape[1] <= 1:
            y_pred = self.lr.predict(self.X)
            plt.scatter(self.X, self.y)
            plt.plot(self.X, y_pred, c='r')
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
        mse = MSE(y_pred=y_pred,y_true=self.y)
        r_score = r2_score(y_true=self.y,y_pred=y_pred)
        print()
        print('MSE:',format(mse))
        print('R2:',format(r_score))
        print()

if __name__ == "__main__":
    #载入数据
    np.random.seed(0)
    x = np.random.uniform(-3, 3, size=100)  # 生成-3到3的100个随机数
    X = x.reshape(-1, 1)  # 变成二维数组
    y = 0.5 * x**2 + 2 + np.random.normal(0, 1, size=100)  # 加上噪音
    reg = linear_regression(X,y)
    reg.__2Ddraw__()
    reg.__parameter__()
    reg.__Lasso__(0.25)
    reg.__evaluate__()
