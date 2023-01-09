import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn import datasets

class logistics_classfier():
    '''
        logistics分类
        X:特征样本
        y:特征标签
        L:正则化方式，可选'l1','l2'
        C:惩罚参数,数值越小正则化越强
    '''
    def __init__(self,X,y,l = 'l2',C = 2):
        self.X = X
        self.y = y
        self.l = l
        self.C = C

    def __2Ddraw__(self):
        plt.scatter(self.X[:, 0],self.X[:, 1],marker='o',c=self.y)
        plt.show()

    def __parameter__(self):
        '''选择参数'''
        #评价拟合效果
        l1_tr = []
        l1_te = []
        l2_tr = []
        l2_te = []
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(self.X, self.y, test_size=0.3, random_state=20)
        #表现
        for i in np.linspace(0.05, 2, 38):
            l1_lr = LogisticRegression(penalty='l1', C=i,solver='liblinear',max_iter=1000).fit(Xtrain, Ytrain)
            l2_lr = LogisticRegression(penalty='l2', C=i,solver='liblinear',max_iter=1000).fit(Xtrain, Ytrain)
            # 比较L1正则化和L2正则化的模型效果差异
            l1_tr.append(accuracy_score(l1_lr.predict(Xtrain), Ytrain))
            l1_te.append(accuracy_score(l1_lr.predict(Xtest), Ytest))
            l2_tr.append(accuracy_score(l2_lr.predict(Xtrain), Ytrain))
            l2_te.append(accuracy_score(l2_lr.predict(Xtest), Ytest))
        graph = [l1_tr, l2_tr, l1_te, l2_te]
        color = ['green', 'black', 'lightgreen', 'gray']
        label = ['l1_train', 'l2_train', 'l1_test', 'l2_test']
        plt.figure()
        for i in range(len(graph)):
            plt.plot(np.linspace(0.05, 2, 38), graph[i], color[i], label=label[i])
        plt.legend(loc=4)
        plt.show()

    def __logistics__(self,test_size = 0.3):
        '''
            logistcs回归
            test_size: 测试集占比
        '''
        #划分测试集
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=0)
        X_combined = np.vstack((X_train, X_test))
        y_combined = np.hstack((y_train, y_test))
        #分类
        if self.l == 'l1':
            solver = 'liblinear'
        else:
            solver = 'lbfgs'
        lr = LogisticRegression(penalty=self.l, C=self.C,solver=solver).fit(X_train, y_train)
        self.lr = lr
        #可能的画图
        NX = np.array(self.X)
        if NX.shape[1]<=2:
            plot_decision_regions(X_combined, y_combined, clf=lr)
            plt.legend(loc='upper left')
            plt.show()
        #参数
        w, b = lr.coef_[0], lr.intercept_
        print('Weight={}\nbias={}'.format(w, b))
        return w,b

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
        report = classification_report(self.y,y_pred)
        print('预测准确度:',acc)
        print(report)

if __name__ == "__main__":
    # 导入数据集
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    clf = logistics_classfier(X,y,'l2',1.25)
    #clf.__2Ddraw__()
    #clf.__parameter__()
    clf.__logistics__()
    clf.__evaluate__()