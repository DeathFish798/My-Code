import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as LR
from sklearn.preprocessing import PolynomialFeatures as pl
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score


class poly_regression():
    '''
        多项式回归
        X:特征样本
        y:特征标签
        d:多项式次数
    '''
    def __init__(self,X,y,d):
        self.X = X
        self.y = y
        self.d = d

    def __2Ddraw__(self):
        plt.scatter(self.X,self.y,marker='o')
        plt.show()

    def __PL__(self):
        '''多项式回归'''
        #回归
        lr = LR()
        pf = pl(degree=self.d)
        lr.fit(pf.fit_transform(self.X),self.y)
        #获取参数
        self.lr = lr
        self.pf = pf
        self.coef = lr.coef_
        self.intercept = lr.intercept_
        print()
        print("w:",self.coef)
        print("b:",self.intercept)
        print()
        #可能的画图
        NX = np.array(self.X)
        if NX.shape[1] <= 1:
            xx = np.linspace(int(min(self.X))-1,int(max(self.X))+1)
            xx2 = pf.transform(xx[:,np.newaxis])
            y_pred = self.lr.predict(xx2)
            plt.scatter(self.X,self.y)
            plt.plot(xx,y_pred,c='r')
            plt.show()

    def __predict__(self,X):
        '''结果预测'''
        X = np.array(X)
        Y = self.lr.predict(self.pf.fit_transform(X))
        print("预测值y:",Y)
        return Y

    def __evaluate__(self):
        '''结果评价'''
        y_pred = self.lr.predict(self.pf.fit_transform(self.X))
        mse = MSE(y_pred=y_pred,y_true=self.y)
        r_score = r2_score(y_true=self.y,y_pred=y_pred)
        print()
        print('MSE:',format(mse))
        print('伪R2:',format(r_score))
        print()

if __name__ == "__main__":
    #载入数据
    np.random.seed(0)
    x = np.random.uniform(-3, 3, size=100)  # 生成-3到3的100个随机数
    X = x.reshape(-1, 1)  # 变成二维数组
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)  # 定义y=0.5x^2+x+2的函数，加上噪音
    reg = poly_regression(X,y,2)
    reg.__2Ddraw__()
    reg.__PL__()
    reg.__evaluate__()
