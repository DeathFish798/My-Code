import numpy as np
import pandas as pd
import gurobipy as gb
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
# coding=utf-8

def Saveexcel(data,filename):
    data = pd.DataFrame(data)
    writer = pd.ExcelWriter(filename)
    data.to_excel(writer, "page_1", float_format='%f')
    writer.save()

if __name__ == "__main__":
    #读取
    point = pd.read_excel("name.xlsx",header=None)
    point = np.array(point)
    print(point)

    a = np.zeros([len(point),1])
    b = np.zeros([len(point),1])
    c = np.zeros([len(point),1])
    for i in range(len(point)):
        a[i,0] = 0 - point[i,1]
        b[i,0] = 0 - point[i,2]
        c[i,0] = 0 - point[i,3]
    com = np.append(a,b,axis = 1)
    com = np.append(com,c,axis = 1)
    #Saveexcel(com,"data1.xlsx")

    #####进行最优化#####
    min = 99999
    step = 0
    setz = []
    for s in range(1201):
        dz = -0.6 + 0.001*s
        t = np.zeros([len(point)-1,1])
        d = np.zeros([len(point)-1,1])
        sum = 0
        for i in range(len(point)-1):
            x0 = -com[i+1,0]
            y0 = -com[i+1,1]
            z0 = -com[i+1,2]
            t[i,0] = (2804*z0 - 20*dz*z0 + 5*((16*(5*dz - 701)*(5*dz*x0**2 + 5*dz*y0**2 + 5*dz*z0**2 - 1502*x0**2 - 1502*y0**2 - 701*z0**2))/25)**(1/2))/(10*(x0**2 + y0**2))
            xj = x0*t[i,0]
            yj = y0*t[i,0]
            zj = z0*t[i,0]
            d[i] = math.sqrt((x0-xj)**2 + (y0 - yj)**2 + (z0-zj)**2)
            if (d[i] - 0.6)>0:
                d[i] = d[i] - 0.6
            else:
                d[i] = 0
            sum = sum + d[i]


        if sum < min:
            min = sum
            step = s
            setz.append(-0.6 + 0.001*step)

    print("dz大小为：",-0.6 + 0.001*step)
    print(min)
    #print(setz)