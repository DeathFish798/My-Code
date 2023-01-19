import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


class _PCA_():
    '''
        X:test
        y:label
    '''
    def __init__(self,X,y = None):
        self.X = X
        self.y = y

    def n_choice(self):
        pca = PCA(n_components=None).fit(self.X)
        evr = pca.explained_variance_ratio_ * 100
        cov_matritx = pca.get_covariance()
        eigenvalue, featurevector = np.linalg.eig(cov_matritx)
        # weight picture
        feat = featurevector[:7,:]
        count = 0
        for i in feat:
            x = range(1,self.X.shape[1] + 1)
            plt.figure()
            plt.bar(x,i,color = 'royalblue')
            plt.ylim(-1,1)
            plt.xticks(range(1,self.X.shape[1] + 1))
            plt.savefig("PCA{}.png".format(str(count)))
            plt.close()
            count += 1
        fig, ax = plt.subplots()
        ax.plot(np.arange(1, len(evr) + 1), np.cumsum(evr), "-bo")
        ax.set_xlabel("num_of_choice")
        ax.set_ylabel("explained_variance_ratio(%)")
        plt.savefig("choice_n.png")
        num = 0
        time = 0
        for i in np.cumsum(evr):
            num = num + 1
            if i >= 85 and time == 0:
                print("number is {}，explained_variance_ratio is {}%".format(num, i))
                time = time + 1
            if i >= 90 and time == 1:
                print("number is {}，explained_variance_ratio is {}%".format(num, i))
                time = time + 1
            if i >= 95 and time == 2:
                print("number is {}，explained_variance_ratio is {}%".format(num, i))
                break

    def PCA(self,target = 0.95):
        ppp = PCA(n_components = target).fit(self.X)
        res = ppp.transform(self.X)
        self.pr = ppp
        print()
        print("initial matrix", self.X.shape)
        print("PCA matrix", res.shape)
        cov_matritx = ppp.get_covariance()
        Saveexcel(cov_matritx, "cov_matrix.xlsx")
        print()
        return res

    def transor(self,X):
        res = self.pr.transform(X)
        return res

class svm_classifier():
    '''
        svm支持向量机
        X:特征样本
        y:特征标签
        C:惩罚参数
        kernal:核函数,线性可分'linear',线性不可分'rbf'
    '''
    def __init__(self,X,y,C = 5,kernel = 'rbf'):
        self.X = X
        self.y = y
        self.C = C
        self.k = kernel

    def __2Ddraw__(self):
        plt.scatter(self.X[:, 0],self.X[:, 1],marker='o',c=self.y)
        plt.show()

    def __parameter__(self,lb=0.5,ub=10,num=19):
        '''选择参数'''
        #划分训练集
        l1_tr = []
        l1_te = []
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(self.X, self.y, test_size=0.3, random_state=20)
        #表现
        for i in np.linspace(lb,ub,num):
            l1_lr = SVC(C=i,kernel=self.k,max_iter=1000).fit(Xtrain, Ytrain)
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

class RFC_classifier():
    '''
        随机森林
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
        plt.scatter(self.X[:, 0], self.X[:, 1], marker='o', c=self.y)
        plt.show()

    def __nstimators__(self, low=10, up=100, step=10):
        '''交叉验证选取树的数量'''
        scorel = []
        for i in range(low, up, step):
            rfc = RandomForestClassifier(n_estimators=i, random_state=20, n_jobs=-1)
            score = cross_val_score(rfc, self.X, self.y, cv=3).mean()
            scorel.append(score)
        print("最高准确率为{}, 树的个数为{}".format(max(scorel), (scorel.index(max(scorel)) * step + low)))
        plt.figure()
        plt.plot(range(low, up, step), scorel)
        plt.show()

    def __RFC__(self, test_size=0.3):
        '''
            随机森林分类
            test_size: 测试集占比
        '''
        # 划分测试集
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=0)
        X_combined = np.vstack((X_train, X_test))
        y_combined = np.hstack((y_train, y_test))
        # 分类
        lr = RandomForestClassifier(n_estimators=self.n_est, max_depth=self.depth, random_state=20).fit(X_train,y_train)
        self.lr = lr

    def __predict__(self, X):
        '''结果预测'''
        X = np.array(X)
        Y = self.lr.predict_proba(X)
        print("预测值y:", Y)
        return Y

    def __evaluate__(self):
        '''结果评价'''
        y_pred = self.lr.predict(self.X)
        acc = accuracy_score(y_true=self.y, y_pred=y_pred)
        report = classification_report(self.y, y_pred)
        print('预测准确度:', acc)
        print(report)

def _3Ddraw_(X):
    fig1 = plt.figure()
    ax1 = plt.axes(projection='3d')
    ax1.scatter(X[:,0],X[:,1],X[:,2])
    plt.show()

def Saveexcel(data,filename):
    '''
        save excel
    '''
    data = pd.DataFrame(data)
    writer = pd.ExcelWriter(filename)
    data.to_excel(writer, "page_1",float_format="%.4f")
    writer.save()

def mean_artist(data):
    res = []
    num = len(data)
    for i in range(data.shape[1]):
        sum = 0
        for j in range(len(data)):
            sum += data[j][i]
        res.append(sum / num)
    return np.array(res)

def euclid(data1,data2):
    sum = 0
    for i in range(len(data1)):
        d = data1[i] - data2[i]
        sum += (d ** 2)
    res = sum ** 0.5
    return res

def similarity(mean,data):
    res = []
    for i in data:
        d = euclid(mean,i)
        res.append(d)
    return np.array(res)

if __name__ == "__main__":
    sheet1 = pd.read_csv("artist.csv")
    data1 = np.array(sheet1)
    print(data1)
    #count artist
    count = 1
    num0 = 74
    for i in range(len(data1)):
        if data1[i][0] != num0:
            num0 = data1[i][0]
            count += 1
    print(count)

    infl = []
    num0 = 0
    count = -1
    # influer data
    for i in range(len(data1)):
        if i == 0:
            num0 = data1[i][0]
            count += 1
        elif data1[i][0] != num0:
            inf0 = []
            inf0.append(data1[i - 1][0])
            inf0.append(data1[i - 1][1])
            inf0.append(data1[i - 1][2])
            inf0.append(data1[i - 1][3])
            infl.append(inf0)
            num0 = data1[i][0]
            count += 1
    infl = np.array(infl)
    print(infl)
    #####problem 2#####
    sheet2 = pd.read_csv("data_by_artist.csv")
    data2 = np.array(sheet2)
    #print(data2)
    #print(len(data2))

    #find kind of genre
    artist = []
    genre = []
    train = []
    count = 0
    for i in range(len(infl)):
        if data2[i][1] == int(infl[count][0]):
            art = []
            tra = []
            for j in range(len(data2[i])):
                art.append(data2[i][j])
                if j > 2:
                    tra.append(data2[i][j - 1])
            train.append(tra)
            genre.append(infl[count][2])
            artist.append(art)
            count += 1
        elif data2[i][1] > int(infl[count][0]):
            count += 1
    artist = np.array(artist)
    train = np.array(train)
    genre = np.array(genre)
    print(artist)
    print(train)
    print(genre)
    print(len(artist))

    #num of genre
    gen0 = set()
    num_of_genre = np.zeros([20,1])
    for i in range(len(genre)):
        gen0.add(genre[i])
    genre_num = list(gen0)
    #print(len(genre_num))
    for i in range(len(genre)):
        for j in range(len(genre_num)):
            if genre[i] == genre_num[j]:
                 num_of_genre[j] += 1
    Saveexcel(genre_num,"genre.xlsx")
    Saveexcel(num_of_genre,"genre num.xlsx")

    #####PCA#####
    scaler = MinMaxScaler()
    result = scaler.fit_transform(train)
    dr = _PCA_(result)
    dr.n_choice()
    data_artist = dr.PCA(6)

    # saperate genre
    kind = ['Pop/Rock', 'R&B;', 'Jazz', 'Country']
    pop = []
    rnb = []
    jazz = []
    country = []
    for i in range(len(data_artist)):
        for j in range(len(kind)):
            if genre[i] == kind[j]:
                art0 = []
                for k in range(data_artist.shape[1]):
                    art0.append(data_artist[i][k])
                if j == 0:
                    pop.append(art0)
                elif j == 1:
                    rnb.append(art0)
                elif j == 2:
                    jazz.append(art0)
                else:
                    country.append(art0)
    pop = np.array(pop)
    rnb = np.array(rnb)
    jazz = np.array(jazz)
    country = np.array(country)

    #mean
    pop_mean = mean_artist(pop)
    rnb_mean = mean_artist(rnb)
    jazz_mean = mean_artist(jazz)
    country_mean = mean_artist(country)
    print("------------------------------------------")
    print(pop_mean)
    print(rnb_mean)
    print(jazz_mean)
    print(country_mean)
    print()
    pop_pop = similarity(pop_mean,pop)
    pop_rnb = similarity(pop_mean,rnb)
    pop_jazz = similarity(pop_mean,jazz)
    pop_country = similarity(pop_mean,country)
    Saveexcel(pop_pop, "pop_pop.xlsx")
    Saveexcel(pop_rnb, "pop_rnb.xlsx")
    Saveexcel(pop_jazz, "pop_jazz.xlsx")
    Saveexcel(pop_country, "pop_country.xlsx")

    rnb_pop = similarity(rnb_mean, pop)
    rnb_rnb = similarity(rnb_mean, rnb)
    rnb_jazz = similarity(rnb_mean, jazz)
    rnb_country = similarity(rnb_mean, country)
    Saveexcel(rnb_pop, "rnb_pop.xlsx")
    Saveexcel(rnb_rnb, "rnb_rnb.xlsx")
    Saveexcel(rnb_jazz, "rnb_jazz.xlsx")
    Saveexcel(rnb_country, "rnb_country.xlsx")

    jazz_pop = similarity(jazz_mean, pop)
    jazz_rnb = similarity(jazz_mean, rnb)
    jazz_jazz = similarity(jazz_mean, jazz)
    jazz_country = similarity(jazz_mean, country)
    Saveexcel(jazz_pop, "jazz_pop.xlsx")
    Saveexcel(jazz_rnb, "jazz_rnb.xlsx")
    Saveexcel(jazz_jazz, "jazz_jazz.xlsx")
    Saveexcel(jazz_country, "pop_country.xlsx")

    country_pop = similarity(country_mean, pop)
    country_rnb = similarity(country_mean, rnb)
    country_jazz = similarity(country_mean, jazz)
    country_country = similarity(country_mean, country)
    Saveexcel(country_pop, "country_pop.xlsx")
    Saveexcel(country_rnb, "country_rnb.xlsx")
    Saveexcel(country_jazz, "country_jazz.xlsx")
    Saveexcel(country_country, "country_country.xlsx")

    print("------------------------------------------")

    #####box picture#####
    #pop

    #rnb


    #####problem 3#####
    #separate genre
    label = pd.read_excel("genre_sorted.xlsx")
    label = np.array(label)
    print(label)
    y = []
    for i in range(len(data_artist)):
        for j in range(len(label)):
            if genre[i] == label[j]:
            #if j < 4:
            #    y.append(j)
            #else:
            #    y.append(4)
                y.append(j)
    print(len(y))

    #####logistic#####
    clf = RFC_classifier(data_artist,y,90)
    clf.__nstimators__(10,120,10)
    clf.__RFC__()
    clf.__predict__(data_artist)
    clf.__evaluate__()