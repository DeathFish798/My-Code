import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn import datasets

class RFC_classifier():
    '''
        随机森林
        X:特征样本
        y:特征标签
        n_est:决策树个数
        depth:树的深度
    '''
    def __init__(self,X,y,n_est = 10,depth = None):
        self.X = X
        self.y = y
        self.n_est = n_est
        self.depth = depth

    def __2Ddraw__(self):
        plt.scatter(self.X[:, 0],self.X[:, 1],marker='o',c=self.y)
        plt.show()

    def __nstimators__(self,low = 10,up = 100,step = 10):
        '''交叉验证选取树的数量'''
        scorel = []
        for i in range(low, up, step):
            rfc = RandomForestClassifier(n_estimators=i,random_state=20,n_jobs=-1)
            score = cross_val_score(rfc, self.X, self.y, cv=10).mean()
            scorel.append(score)
        print("最高准确率为{}, 树的个数为{}".format(max(scorel), (scorel.index(max(scorel)) * step + low)))
        plt.figure()
        plt.plot(range(low, up, step), scorel)
        plt.show()

    def __RFC__(self,test_size = 0.3):
        '''
            随机森林分类
            test_size: 测试集占比
        '''
        #划分测试集
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=0)
        X_combined = np.vstack((X_train, X_test))
        y_combined = np.hstack((y_train, y_test))
        #分类
        lr = RandomForestClassifier(n_estimators=self.n_est,max_depth=self.depth,random_state=20).fit(X_train, y_train)
        self.lr = lr
        #可能的画图
        NX = np.array(self.X)
        if NX.shape[1]<=2:
            plot_decision_regions(X_combined, y_combined, clf=lr)
            plt.legend(loc='upper left')
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
        acc = accuracy_score(y_true=self.y, y_pred=y_pred)
        report = classification_report(self.y, y_pred)
        print('预测准确度:', acc)
        print(report)

if __name__ == "__main__":
    # 导入数据集
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    clf = RFC_classifier(X,y,30)
    clf.__nstimators__()
    clf.__RFC__()
    clf.__evaluate__()