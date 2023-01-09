import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets._samples_generator import make_moons
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False

class DBSCAN_cluster():
    '''
        DBSCAN聚类
        X:特征样本
        eps:邻域半径
        minpt:最少点数
    '''
    def __init__(self,X,eps = 0.5,minpt = 0):
        self.X = X
        NX = np.array(X)
        self.eps = eps
        self.k = 2 * NX.shape[1] - 1
        if minpt == 0:
            self.minpt = self.k + 1
        else:
            self.minpt = minpt

    def __2Ddraw__(self):
        plt.scatter(self.X[:, 0],self.X[:, 1], marker='o')  # 假设暂不知道y类别，不设置c=y，使用kmeans聚类
        plt.show()

    def __3Ddraw__(self):
        fig1 = plt.figure()
        ax1 = plt.axes(projection='3d')
        ax1.scatter(self.X[:,0],self.X[:,1],self.X[:,2])
        plt.show()

    def select_MinPts(self,data, k):
        '''选择合适的参数'''
        k_dist = []
        for i in range(data.shape[0]):
            dist = (((data[i] - data) ** 2).sum(axis=1) ** 0.5)
            dist.sort()
            k_dist.append(dist[k])
        return np.array(k_dist)

    def __elbow__(self):
        '''手肘法则确定聚类个数'''
        k_dist = self.select_MinPts(self.X, self.k)
        k_dist.sort()
        plt.plot(np.arange(k_dist.shape[0]), k_dist[::-1])
        eps = k_dist[::-1][15]
        plt.scatter(15, eps, color="r")
        plt.plot([0, 15], [eps, eps], linestyle="--", color="r")
        plt.plot([15, 15], [0, eps], linestyle="--", color="r")
        plt.show()

    def __DBSCAN__(self):
        '''聚类'''
        self.y_pred = DBSCAN(eps=self.eps,min_samples=self.minpt).fit_predict(self.X)
        return self.y_pred

    def __show2D__(self):
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y_pred)
        plt.show()
        print("y_pred:",self.y_pred)

    def __show3D__(self):
        fig2 = plt.figure()
        ax2 = plt.axes(projection='3d')
        ax2.scatter(self.X[:,0],self.X[:,1],self.X[:,2],c=self.y_pred)
        plt.show()
        print("y_pred:",self.y_pred)


if __name__ == "__main__":
    X, y = make_moons(n_samples=1000,noise = 0.05,random_state= 2003)
    cluster = DBSCAN_cluster(X,0.08)
    cluster.__elbow__()
    cluster.__DBSCAN__()
    cluster.__show2D__()
