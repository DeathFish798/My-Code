import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def Saveexcel(data,filename):
    data = pd.DataFrame(data)
    writer = pd.ExcelWriter(filename)
    data.to_excel(writer, "page_1", float_format='%.3f')
    writer.save()

'''
    代码部分
'''
excel_name = '附件1.xlsx'

data_order = pd.read_excel(excel_name, sheet_name=0)
data_supply = pd.read_excel(excel_name, sheet_name=1)

data_order = np.array(data_order)
data_supply = np.array(data_supply)

row_n = data_supply.shape[0]
col_n = data_supply.shape[1]

sig_supply = pd.read_csv('sort1.txt', header = None)
sig_supply = np.array(sig_supply)
sig_supply = np.reshape(sig_supply,[1,50])

means = np.zeros([50,1]) #平均数
sigma = np.zeros([50,1]) #标准差
max_supply = np.zeros([50,1]) #最大供应量
power = np.zeros([50,1]) #产能转化率
money = np.zeros([50,1]) #价格
complet_rate = np.zeros([50,1]) #接单率


new_data_order = []
new_data_supply = []
for i in range(0,50):
    for j in range(row_n):
        if data_order[j][0] == 'S' + str(sig_supply[0,i]).rjust(3,'0'):
            new_data_order.append(data_order[j,:])
            new_data_supply.append(data_order[j,:])

new_data_supply = np.array(new_data_supply)
new_data_order = np.array(new_data_order)


for i in range(0,50):#求平均数及标准差、完成率
    val = 0 #有效数据
    fail = 0 #未接单
    mean = 0 #二次平均值
    for j in range(2, col_n):
        if new_data_order[i][j] == 0:
            continue
        if new_data_supply[i][j] == 0:
           fail = fail + 1
        val = val + 1
        means[i,0] = means[i,0] + new_data_supply[i][j]
    means[i,0] = means[i,0] / val
    complet_rate[i,0] = (val - fail) / val

    for j in range(2, col_n):
        if new_data_order[i][j] == 0:
            continue
        if new_data_supply[i][j] > 1.2 * means[i][0]:
            new_data_supply[i][j] = means[i][0]
        mean = mean + new_data_supply[i][j]
    means[i,0] = mean / val

    sum2 = 0
    for j in range(2, col_n):
        if new_data_order[i][j] == 0:
            continue
        sum2 = (new_data_supply[i][j] - means[i,0])**2 + sum2
    sigma[i,0] = math.sqrt(sum2 / val)

for i in range(50):#筛选最大值
    max = 0
    for j in range(2,col_n):
        if new_data_supply[i,j] > max and new_data_supply[i,j] <= means[i,0] + 3 * sigma[i,0]:
            max = new_data_supply[i][j]
    max_supply[i,0] = max

for i in range(50):#转化为产能
    if new_data_order[i,1] == 'A':
        means[i,0] = means[i,0] / 0.6
        sigma[i,0] = sigma[i,0] / 0.6
        power[i,0] = 1 / 0.6
        money[i,0] = 1.2
    if new_data_order[i,1] == 'B':
        means[i,0] = means[i,0] / 0.66
        sigma[i, 0] = sigma[i, 0] / 0.66
        power[i, 0] = 1 / 0.66
        money[i,0] = 1.1
    if new_data_order[i,1] == 'C':
        means[i,0] = means[i,0] / 0.72
        sigma[i, 0] = sigma[i, 0] / 0.72
        power[i, 0] = 1 /0.72
        money[i,0] = 1



com_supply = np.insert(new_data_supply[:,0:2],2,max_supply.T,axis = 1)
com_supply = np.insert(com_supply,3,power.T,axis = 1)
com_supply = np.insert(com_supply,4,money.T,axis = 1)
com_supply = np.insert(com_supply,5,sigma.T,axis = 1)

print(complet_rate)
#print(com_supply)
#Saveexcel(com_supply,"sort2.xlsx")

