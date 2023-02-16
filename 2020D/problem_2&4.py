import numpy as np
import pandas as pd
from math import *
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


class svm_classifier():
    """
        svm
    """
    def __init__(self,X,y,C = 2,kernel = 'rbf'):
        self.X = X
        self.y = y
        self.C = C
        self.k = kernel

    def __SVC__(self,test_size = 0.3):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=0)
        X_combined = np.vstack((X_train, X_test))
        y_combined = np.hstack((y_train, y_test))
        lr = SVC(C=self.C,kernel=self.k,probability=True).fit(X_train, y_train)
        self.lr = lr
        NX = np.array(self.X)
        if NX.shape[1]<=2:
            plot_decision_regions(X_combined, y_combined, clf=lr)
            plt.legend(loc='upper left')
            plt.show()

    def __predict__(self,X):
        X = np.array(X)
        Y = self.lr.predict(X)
        Yb = self.lr.predict_proba(X)
        print("predict:",Y)
        print("proba:",Yb)
        return Y,Yb

    def __evaluate__(self):
        y_pred = self.lr.predict(self.X)
        acc = accuracy_score(y_true=self.y, y_pred=y_pred)
        report = classification_report(self.y, y_pred)
        print('accuracy:', acc)
        print(report)

def Saveexcel(data,filename):
    '''
        save excel
    '''
    data = pd.DataFrame(data)
    writer = pd.ExcelWriter(filename)
    data.to_excel(writer, "page_1",float_format="%.2f")
    writer.save()

def MinMax(data):
    scaler = MinMaxScaler()
    result = scaler.fit_transform(data)
    return result

if __name__ == "__main__":
    #####read data#####
    matches = pd.read_csv("matches.csv")
    matches = np.array(matches)
    dcdata = []
    cludata = []
    for i in range(19):
        data = pd.read_excel("data_opponent{}.xlsx".format(str(i + 1)))
        data = np.array(data)
        dc = 0
        clu = 0
        for j in range(len(data)):
            dc += data[j,2]
            clu += data[j,3]
        dc = dc / len(data)
        clu = clu / len(data)
        dcdata.append(dc)
        cludata.append(clu)
    print("dc:",dcdata)
    print("clu:",cludata)
    pass_foul_data = pd.read_excel("pass_foul_data.xlsx")
    pass_foul_data = np.array(pass_foul_data)
    passtime = pass_foul_data[:,2]
    foul = pass_foul_data[:,3]
    print(passtime)
    print(foul)

    #####SVM classify#####
    X = list([dcdata,cludata,passtime,foul])
    X = np.array(X).T
    X_init = X
    Saveexcel(X,"X.xlsx")
    X = MinMax(X)

    y = np.zeros(19)
    for i in range(len(matches)):
        for j in range(19):
            if matches[i,1] == "Opponent" + str(j + 1):
                if matches[i,2] == "win":
                    y[j] += 1
                elif matches[i,2] == "loss":
                    y[j] -= 1
    print(y)
    for i in range(len(y)):
        if y[i] > 0:
            y[i] = 2
        elif y[i] < 0:
            y[i] = 0
        else:
            y[i] = 1
    print(y)

    clf = svm_classifier(X,y)
    clf.__SVC__()
    res,proba = clf.__predict__(X)
    Saveexcel(proba,"proba.xlsx")
    clf.__evaluate__()

    #####analysis######
    loss = []
    tie = []
    win = []
    for i in range(len(y)):
        data = []
        for j in range(len(X_init[i])):
            data.append(X_init[i,j])
        if y[i] == 0:
            loss.append(data)
        elif y[i] == 1:
            tie.append(data)
        else:
            win.append(data)
    loss = np.array(loss)
    tie = np.array(tie)
    win = np.array(win)
    Saveexcel(loss,"loss.xlsx")
    Saveexcel(tie,"tie.xlsx")
    Saveexcel(win,"win.xlsx")