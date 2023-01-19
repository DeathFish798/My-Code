import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from sklearn.datasets import make_classification

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False

class _PCA_():
    '''
        主成分分析
        X:样本特征
        y:样本标签
    '''
    def __init__(self,X,y = None):
        self.X = X
        self.y = y

    def n_choice(self):
        pca = PCA(n_components=None).fit(self.X)
        evr = pca.explained_variance_ratio_ * 100
        # 查看累计解释方差比率与主成分个数的关系
        fig, ax = plt.subplots()
        ax.plot(np.arange(1, len(evr) + 1), np.cumsum(evr), "-bo")
        ax.set_xlabel("主成分数量")
        ax.set_ylabel("解释方差比率(%)")
        plt.show()
        num = 0
        time = 0
        for i in np.cumsum(evr):
            num = num + 1
            if i >= 85 and time == 0:
                print("主成分个数为{}，解释方差比率达到%{}".format(num,i))
                time = time + 1
            if i >= 90 and time == 1:
                print("主成分个数为{}，解释方差比率达到%{}".format(num, i))
                time = time + 1
            if i >= 95 and time == 2:
                print("主成分个数为{}，解释方差比率达到%{}".format(num, i))
                break

    def PCA(self,target = 0.95):
        ppp = PCA(n_components = target).fit(self.X)
        res = ppp.transform(self.X)
        self.pr = ppp
        print()
        print("原始矩阵大小：",self.X.shape)
        print("转换矩阵大小",res.shape)
        cov_matritx = ppp.get_covariance()
        print(cov_matritx)
        print()
        return res

if __name__ == "__main__":
    X, y = make_classification(
        n_samples=1000,  # 1000个观测值
        n_features=50,  # 50个特征
        n_informative=10,  # 只有10个特征有预测能力
        n_redundant=40,  # 40个冗余的特征
        n_classes=2,  # 目标变量包含两个类别
        random_state=123  # 随机数种子，保证可重复结果
    )
    dr = _PCA_(X,y)
    dr.n_choice()
    res = dr.PCA(9)
    print(res)
