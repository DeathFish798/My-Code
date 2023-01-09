import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve,auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

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
        pca = PCA(n_components=None).fit(X)
        evr = pca.explained_variance_ratio_ * 100
        cov_matritx = pca.get_covariance()
        eigenvalue, featurevector = np.linalg.eig(cov_matritx)
        print("特征值：", eigenvalue)
        print("特征向量：", featurevector)
        # 绘制权重图
        feat = featurevector[:3,:]
        for i in feat:
            x = range(1,16)
            plt.figure()
            plt.bar(x,i,color = 'royalblue')
            plt.ylim(-1,1)
            plt.xticks(range(1,16))
            plt.show()

        # 查看累计解释方差比率与主成分个数的关系
        fig, ax = plt.subplots()
        ax.plot(np.arange(1, len(evr) + 1), np.cumsum(evr), "-bo")
        ax.set_xlabel("主成分数量")
        ax.set_ylabel("解释方差比率(%)")
        plt.savefig("主成分分析参数选择.png")
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
        Saveexcel(cov_matritx,"协方差矩阵.xlsx")
        print()
        return res

    def transor(self,X):
        res = self.pr.transform(X)
        return res

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
            l1_tr.append(accuracy_score(l1_lr.predict(Xtrain), Ytrain))
            l1_te.append(accuracy_score(l1_lr.predict(Xtest), Ytest))
            l2_tr.append(accuracy_score(l2_lr.predict(Xtrain), Ytrain))
            l2_te.append(accuracy_score(l2_lr.predict(Xtest), Ytest))
        g = [l1_tr, l2_tr, l1_te, l2_te]
        c = ['green', 'black', 'lightgreen', 'gray']
        lab = ['l1_train', 'l2_train', 'l1_test', 'l2_test']
        plt.figure()
        for i in range(len(g)):
            plt.plot(np.linspace(0.05, 2, 38), g[i], c[i], label=lab[i])
        plt.legend(loc=4)
        plt.savefig("logistics参数选择.png")

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
        fpr, tpr, thresholds = roc_curve(self.y, y_pred)
        print("auc:",auc(fpr,tpr))
        print('预测准确度:',acc)
        plt.figure()
        roc_auc = auc(fpr,tpr)
        plt.title('ROC曲线')
        plt.plot(fpr, tpr, label=u'AUC = %0.3f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.ylabel('TP Rate')
        plt.xlabel('FP Rate')
        plt.grid(linestyle='-.')
        plt.grid(True)
        plt.savefig("ROC曲线")
        print(roc_auc)
        print(report)

class RFC_classifier():
    '''
        随机森林
        X:特征样本
        y:特征标签
        n_est:决策树个数
        depth:树的深度
    '''
    def __init__(self,X,y,n_est = 10,depth = 10):
        self.X = X
        self.y = y
        self.n_est = n_est
        self.depth = depth

    def __nstimators__(self,low = 10,up = 100,step = 10):
        '''交叉验证选取树的数量'''
        sco = []
        for i in range(low, up, step):
            rfc = RandomForestClassifier(n_estimators=i,random_state=20,n_jobs=-1)
            score = cross_val_score(rfc, self.X, self.y, cv=10).mean()
            sco.append(score)
        print("最高准确率为{}".format(max(sco), (sco.index(max(sco)) * step + low)))
        plt.figure()
        plt.plot(range(low, up, step), sco)
        plt.savefig("随机森林参数选择.png")

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
        fpr, tpr, thresholds = roc_curve(self.y, y_pred)
        print("auc:", auc(fpr, tpr))
        print('预测准确度:', acc)
        print(report)

def Saveexcel(data,filename):
    '''
        保存excel
    '''
    data = pd.DataFrame(data)
    writer = pd.ExcelWriter(filename)
    data.to_excel(writer, "page_1", float_format='%.2f')
    writer.save()

def acc(y_true,y_pred):
    right = 0
    wrong = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            right = right + 1
        else:
            wrong = wrong + 1
    return right / (right + wrong)

if __name__ == "__main__":
    #####数据读取#####
    data = pd.read_excel("问题一合并数据.xlsx")
    X = np.array(data.iloc[:,2:-1])
    y = np.array(data.iloc[:,-1])
    print(X)
    print(y)

    #####主成分分析#####
    dr = _PCA_(X)
    dr.n_choice()
    new_X = dr.PCA(3)
    Saveexcel(new_X,"主成分.xlsx")

    #####分类#####
    # 随机森林

    clf2 = RFC_classifier(new_X, y)
    clf2.__nstimators__(1,10,1)
    clf2.n_est = 5 #选择树的个数
    clf2.__RFC__()
    clf2.__evaluate__()

    clf4 = RFC_classifier(new_X, y)
    clf4.__nstimators__(1, 10, 1)
    clf4.n_est = 1  # 选择树的个数
    clf4.__RFC__()
    clf4.__evaluate__()

    #逻辑斯蒂
    clf1 = logistics_classfier(new_X, y)
    clf1.__parameter__()
    clf1.__logistics__(0.7) #训练集占比0.7
    clf1.l = 'l1' #选择l1正则化
    clf1.C = 1.75
    clf1.__evaluate__()

    #验证
    clf3 = logistics_classfier(new_X, y)
    clf3.__logistics__(0.5) #训练集占比0.5
    clf3.l = 'l1'
    clf3.C = 1.75
    clf3.__evaluate__()

    #####结果预测#####
    b3 = pd.read_excel("b3.xlsx")
    b3 = np.array(b3.iloc[:,1:])

    #处理数据
    for i in range(len(b3)):
        for j in range(b3.shape[1]):
            if j == 0:
                if b3[i,j] == "无风化":
                    b3[i,j] = 0
                else:
                    b3[i,j] = 1
            else:
                if not b3[i,j] > 0:
                    b3[i,j] = 0
    print(b3.shape)
    test = np.zeros([len(b3),b3.shape[1]])
    for i in range(len(test)):
        for j in range(test.shape[1]):
            if j < test.shape[1] - 1:
                test[i,j] = b3[i,j + 1]
            else:
                test[i,j] = b3[i,0]
    b3_test = test

    #降维
    test = dr.transor(test)
    print(test.shape)

    #随机森林预测
    res_rf1 = clf2.__predict__(test)
    res_rf2 = clf4.__predict__(test)

    #logistics预测
    res_lr1 = clf1.__predict__(test)
    res_lr2 = clf3.__predict__(test)

    #####问题二敏感性分析#####
    print("第二题敏感度分析")
    y_true = y
    loser = []
    xser = []
    np.random.seed(20)
    for i in range(20):
        u = i + 1
        eps = np.random.uniform(-u/2,u/2,(X.shape))
        epX = X + eps
        # 主成分
        ep_test = dr.transor(epX)
        #逻辑斯蒂
        y_pred = clf1.__predict__(ep_test)
        # 准确率
        loss = acc(y_true, y_pred)
        loser.append(loss)
        xser.append(u/2)
    #可视化
    plt.figure()
    plt.plot(xser,loser)
    plt.xticks(xser)
    plt.ylim(0,1.1)
    plt.xticks(rotation=300,fontsize=8)
    plt.xlabel("噪声强度")
    plt.tight_layout()
    plt.savefig("问题二logistic敏感性分析.png")

    #####问题三敏感性分析#####
    print("问题三敏感性分析")
    y_true = [0,1,1,1,1,0,0,1]
    loser = []
    xser = []
    np.random.seed(20)
    for i in range(20):
        u = i + 1
        eps = np.random.uniform(-u/2,u/2,(b3_test.shape))
        epX = b3_test + eps
        # 主成分
        ep_test = dr.transor(epX)
        # 逻辑斯蒂
        y_pred = clf1.__predict__(ep_test)
        # 准确率
        loss = acc(y_true, y_pred)
        loser.append(loss)
        xser.append(u/2)
    #可视化
    plt.figure()
    plt.plot(xser,loser)
    plt.xticks(xser)
    plt.ylim(0,1.1)
    plt.xticks(rotation=300,fontsize=10)
    plt.xlabel("噪声强度")
    plt.tight_layout()
    plt.savefig("问题三logistic敏感性分析.png")



