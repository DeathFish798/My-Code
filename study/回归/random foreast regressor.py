import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score

class RFR_regression():
    '''
        随机森林回归
        X:特征样本
        y:特征标签
        n_est:决策树个数
        depth:树的深度
    '''

    def __init__(self, X, y, n_est=10, depth=None):
        self.X = X
        self.y = y
        self.n_est = n_est
        self.depth = depth

    def __2Ddraw__(self):
        plt.scatter(self.X,self.y,marker='o')
        plt.show()

    def __nstimators__(self,low = 10,up = 100,step = 10):
        '''交叉验证选取树的数量'''
        scorel = []
        for i in range(low, up, step):
            rfr = RFR(n_estimators=i,random_state=20,n_jobs=-1)
            score = cross_val_score(rfr, self.X, self.y, scoring='neg_mean_squared_error',cv=10, n_jobs=-1).mean()
            scorel.append(score)
        print("最低MSE为{}, 树的个数为{}".format(-max(scorel), (scorel.index(max(scorel)) * step + low)))
        plt.figure()
        plt.plot(range(low, up, step), scorel)
        plt.show()

    def __RFR__(self):
        '''多项式回归'''
        #回归
        lr = RFR(n_estimators=self.n_est,max_depth=self.depth).fit(self.X,self.y)
        self.lr = lr
        #可能的画图
        NX = np.array(self.X)
        if NX.shape[1] <= 1:
            xx = np.linspace(int(min(self.X)) - 1, int(max(self.X)) + 1)
            xx = xx.reshape(-1, 1)
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
        mse = MSE(y_pred=y_pred, y_true=self.y)
        r_score = r2_score(y_true=self.y, y_pred=y_pred)
        print()
        print('MSE:', format(mse))
        print('伪R2:', format(r_score))
        print()
if __name__ == "__main__":
    #载入数据
    np.random.seed(0)
    x = np.random.uniform(-3, 3, size=100)  # 生成-3到3的100个随机数
    X = x.reshape(-1, 1)  # 变成二维数组
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)  # 定义y=0.5x^2+x+2的函数，加上噪音
    reg = RFR_regression(X,y)
    reg.__2Ddraw__()
    reg.__nstimators__(100,500,100)
    reg.n_est = 300
    reg.__RFR__()
    reg.__evaluate__()