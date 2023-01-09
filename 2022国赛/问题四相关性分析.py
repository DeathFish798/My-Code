import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False

def Saveexcel(data,filename):
    '''
        保存excel
    '''
    data = pd.DataFrame(data)
    writer = pd.ExcelWriter(filename)
    data.to_excel(writer, "page_1", float_format='%.2f')
    writer.save()

if __name__ == "__main__":
    #####数据读取#####
    c1c2 = pd.read_excel("c1mc2.xlsx")
    c3c4 = pd.read_excel("c3mc4.xlsx")

    title = list(c1c2)
    title = np.array(title)
    print(title)

    c1c2 = np.array(c1c2)
    c3c4 = np.array(c3c4)

    print(c1c2)
    print(c3c4.shape)

    #####计算相关系数#####
    cof1 = np.corrcoef(c1c2.T)
    cof2 = np.corrcoef(c3c4.T)
    print(cof1.shape)
    print(cof2.shape)
    Saveexcel(cof1,"高钾相关系数.xlsx")
    Saveexcel(cof2,"铅钡相关系数.xlsx")

    #####绘制热力图#####
    plt.figure(figsize=(20,15),dpi=50)
    ax = sns.heatmap(cof1,cmap="RdBu_r")
    plt.xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5], title)
    plt.yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5], title)
    plt.xticks(rotation=300,fontsize=15)
    plt.yticks(rotation=300,fontsize=15)
    plt.savefig("高钾相关系数.png")

    plt.figure(figsize=(20, 15), dpi=50)
    ax = sns.heatmap(cof2, cmap="RdBu_r")
    plt.xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5], title)
    plt.yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5], title)
    plt.xticks(rotation=300, fontsize=15)
    plt.yticks(rotation=300, fontsize=15)
    plt.savefig("铅钡相关系数.png")


