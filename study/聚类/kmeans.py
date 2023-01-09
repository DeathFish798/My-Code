import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn.metrics import silhouette_score

from sklearn.datasets import make_blobs

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False

class kmeans_cluster():
    '''
        kmeans聚类
        X:特征样本
        k:类别数量
    '''
    def __init__(self,X,k=3):
        self.X = X
        self.k = k

    def __2Ddraw__(self):
        plt.scatter(self.X[:, 0],self.X[:, 1], marker='o')
        plt.show()

    def __3Ddraw__(self):
        fig1 = plt.figure()
        ax1 = plt.axes(projection='3d')
        ax1.scatter(self.X[:,0],self.X[:,1],self.X[:,2])
        plt.show()

    def __kmeans__(self):
        '''聚类'''
        self.y_pred = KMeans(n_clusters=self.k,random_state=30).fit_predict(self.X)
        return self.y_pred

    def __kmeanspp__(self):
        '''kmeans++聚类'''
        self.y_pred = KMeans(n_clusters=self.k,init='k-means++').fit_predict(self.X)
        return self.y_pred

    def __MiniBatchKMeans__(self,batch_size = 100):
        '''minibatch-kmeans聚类'''
        self.y_pred = MiniBatchKMeans(n_clusters=self.k,batch_size=batch_size).fit_predict(self.X)
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

    def __elbow__(self,min = 2,max = 9):
        '''手肘法则确定聚类个数'''
        SSE = []  # 误差平方和
        index = [] # 轮廓系数
        for t in range(min,max): #k的迭代范围
            estimator = KMeans(n_clusters=t)  # 构造聚类器
            estimator.fit(self.X)
            SSE.append(estimator.inertia_) # estimator.inertia_获取聚类准则的总和

            y_est = KMeans(n_clusters=t).fit_predict(self.X)
            value = silhouette_score(self.X, y_est)
            index.append(value)
        t = range(min,max)
        plt.xlabel('k')
        plt.ylabel('SSE')
        plt.plot(t,SSE,'o-')
        plt.title("手肘法则")
        plt.show()

        plt.xlabel('k')
        plt.ylabel('index')
        plt.plot(t,index,'ro-')
        plt.title("轮廓系数")
        plt.show()



if __name__ == "__main__":
    X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1, -1,-1], [0, 0,0], [1, 1,1], [2, 2,2]],
                      cluster_std=[0.4, 0.2, 0.2, 0.2], random_state=3)
    cluster = kmeans_cluster(X,4)
    cluster.__3Ddraw__()
    cluster.__elbow__()
    y_pred = cluster.__kmeans__()
    cluster.__show3D__()

