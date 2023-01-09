import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn import datasets

class svm_classifier():
    '''
        svm支持向量机
        X:特征样本
        y:特征标签
        C:惩罚参数
        kernal:核函数,线性可分'linear',线性不可分'rbf'
    '''
    def __init__(self,X,y,C = 2,kernel = 'rbf'):
        self.X = X
        self.y = y
        self.C = C
        self.k = kernel

    def __2Ddraw__(self):
        plt.scatter(self.X[:, 0],self.X[:, 1],marker='o',c=self.y)
        plt.show()

    def __parameter__(self,lb=0.1,ub=2,num=19):
        '''选择参数'''
        #划分训练集
        l1_tr = []
        l1_te = []
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(self.X, self.y, test_size=0.3, random_state=20)
        #表现
        for i in np.linspace(lb,ub,num):
            l1_lr = SVC(C=i,kernel=self.k).fit(Xtrain, Ytrain)
            l1_tr.append(accuracy_score(l1_lr.predict(Xtrain), Ytrain))
            l1_te.append(accuracy_score(l1_lr.predict(Xtest), Ytest))
        graph = [l1_tr, l1_te]
        color = ['green', 'lightgreen']
        label = ['train', 'test']
        plt.figure()
        for i in range(len(graph)):
            plt.plot(np.linspace(lb,ub,num), graph[i], color[i], label=label[i])
        plt.legend(loc=4)
        plt.title('C')
        plt.show()

    def __SVC__(self,test_size = 0.3):
        '''
            SVC分类
            test_size: 测试集占比
        '''
        #划分测试集
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=0)
        X_combined = np.vstack((X_train, X_test))
        y_combined = np.hstack((y_train, y_test))
        #分类
        lr = SVC(C=self.C,kernel=self.k).fit(X_train, y_train)
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

    clf = svm_classifier(X,y,0.2)
    clf.__2Ddraw__()
    clf.__parameter__()
    clf.__SVC__()
    V = np.array([[2, 3],
                  [1, 1]])
    clf.__predict__(V)
    clf.__evaluate__()
