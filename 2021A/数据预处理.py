import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

def Saveexcel(data,filename):
    data = pd.DataFrame(data)
    writer = pd.ExcelWriter(filename)
    data.to_excel(writer, "page_1", float_format='%f')
    writer.save()

if __name__ == "__main__":
    #读取
    main_point = pd.read_csv("1.csv",header=None)
    drive = pd.read_csv("2.csv",header=None)
    face = pd.read_csv("3.csv",header=None)

    main_point = np.array(main_point)
    drive = np.array(drive)
    face = np.array(face)

    #print(face)
    print(main_point)
    '''
    #####图一#####
    # 绘制主节点
    fig1 = plt.figure()
    ax1 = plt.axes(projection = '3d')

    ax1.set_zlim(-350,0)
    for i in range(len(main_point)):
        x = main_point[i,1]
        y = main_point[i,2]
        z = main_point[i,3]
        color = 'orange'
        num = main_point[i,0]
        if num[0] == 'A':
            color = 'red'
        elif num[0] == 'C':
            color = 'yellow'
        elif num[0] == 'D':
            color = 'green'
        elif num[0] == 'E':
            color = 'blue'
        ax1.trisurf(x,y,z,c = color,s = 2)
    
    #绘制促动器

    for i in range(len(drive)):
        xx = [drive[i,1]]
        yy = [drive[i,2]]
        zz = [drive[i,3]]
        xx.append(drive[i,4])
        yy.append(drive[i,5])
        zz.append(drive[i,6])
        color = 'grey'
        ax1.plot3D(xx, yy, zz, c=color, marker='x', ms = 2)
    

    #####图二#####
    #绘制表面

    fig2 = plt.figure()
    ax2 = plt.axes(projection = '3d')
    ax2.set_zlim(-350, 0)
    for i in range(len(face)):
        n1 = face[i,0]
        n2 = face[i,1]
        n3 = face[i,2]
        xx = []
        yy = []
        zz = []
        for j in range(len(main_point)):
            if n1 == main_point[j,0]:
                xx.append(main_point[j,1])
                yy.append(main_point[j,2])
                zz.append(main_point[j,3])
            if n2 == main_point[j,0]:
                xx.append(main_point[j,1])
                yy.append(main_point[j,2])
                zz.append(main_point[j,3])
            if n3 == main_point[j,0]:
                xx.append(main_point[j,1])
                yy.append(main_point[j,2])
                zz.append(main_point[j,3])
        xx.append(xx[0])
        yy.append(yy[0])
        zz.append(zz[0])
        ax2.plot3D(xx,yy,zz,c = 'royalblue',ms = 2)


    #####图三#####
    #绘制促动器
    fig3 = plt.figure()
    ax3 = plt.axes(projection='3d')
    ax3.set_zlim(-350, 0)
    for i in range(len(drive)):
        xx = [drive[i,1]]
        yy = [drive[i,2]]
        zz = [drive[i,3]]
        xx.append(drive[i,4])
        yy.append(drive[i,5])
        zz.append(drive[i,6])
        color = 'grey'
        ax3.plot3D(xx, yy, zz, c=color, marker='x', ms = 2)

    #plt.show()
    plt.savefig("fig1.png")
    plt.close()
    plt.savefig("fig2.png")
    plt.close()
    plt.savefig("fig3.png")
    plt.close()
    '''
    #####计算距离#####
    #原点距离
    r = 0
    for i in range(len(main_point)):
        k = 0
        k = main_point[i,1] ** 2 + main_point[i,2] ** 2 + main_point[i,3] ** 2
        k = math.sqrt(k)
        r = r + k
    r = r / len(main_point)
    print("平均半径为：",r)

    #####找照明区域内点#####
    #寻找下方点
    low_z = r**2 - 150 ** 2
    low_z = math.sqrt(low_z)
    print("最低高度为：",low_z)
    need_point = []

    for i in range(len(main_point)):
        if main_point[i,3] < -low_z:
            need_point.append([main_point[i,0],main_point[i,1],main_point[i,2],main_point[i,3]])
    print("第一次找到的点数：",len(need_point))

    #寻找连接点
    set = {'apple'}
    set.pop()
    for i in need_point:
        set.add(i[0])
    set.add('B1')
    for i in set.copy():
        for j in range(len(face)):
            if i == face[j,0]:
                set.add(face[j,1])
                set.add(face[j,2])


    #返回需求点
    new_need = np.zeros([len(set),3])
    need_name = []
    val = 0
    for i in set:
        for j in range(len(main_point)):
            if main_point[j,0] == str(i):
                need_name.append(main_point[j,0])
                new_need[val, 0] = main_point[j, 1]
                new_need[val, 1] = main_point[j, 2]
                new_need[val, 2] = main_point[j, 3]
                break
        val = val + 1
    need_name = np.array(need_name)
    print("共找到{}个点".format(len(need_name)))
    print(need_name)
    print(new_need)

    #Saveexcel(need_name,"name.xlsx")
    #Saveexcel(new_need,"need.xlsx")

    #####求下拉索长度#####
    below_lenth = np.zeros([len(main_point),1])
    sum_lenth = 0
    for i in range(len(main_point)):
        x1 = main_point[i,1]
        x2 = drive[i,4]
        y1 = main_point[i,2]
        y2 = drive[i,5]
        z1 = main_point[i,3]
        z2 = drive[i,6]
        long = math.sqrt((x1-x2)**2 + (y1-y2)**2 +(z1-z2)**2)
        below_lenth[i,0] = long
        sum_lenth = sum_lenth + long
    print(below_lenth)
    print("平均长度为：",sum_lenth/len(below_lenth))
    Saveexcel(below_lenth,"下拉索长度.xlsx")

