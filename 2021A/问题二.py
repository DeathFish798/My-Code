import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

def Saveexcel(data,filename):
    data = pd.DataFrame(data)
    writer = pd.ExcelWriter(filename)
    data.to_excel(writer, "page_1", float_format='%.3f')
    writer.save()

def retran(a,b,c):
    alpha = math.radians(36.795)
    beta = math.radians(78.169)
    r = 300.4

    #x = -(r*math.cos(beta))/math.sqrt(1+math.tan(alpha)**2)
    #y = -r*math.cos(beta)*math.tan(alpha)
    #z = -r*math.sin(beta)
    #print(x,y,z)
    #R = np.array([[-z*math.cos(alpha)/r,-math.sin(alpha),-x/r],
    #              [z*math.sin(alpha)/r,math.cos(alpha),-y/r],
    #              [-((x*math.cos(alpha)+y*math.sin(alpha))/r),0,-z/r]])
    A = np.array([
        [math.sin(beta),0,-math.cos(beta)],
        [0,1,0],
        [math.cos(beta),0,math.sin(beta)]
    ])
    B = np.array([
        [math.cos(alpha),math.sin(alpha),0],
        [-math.sin(alpha),math.cos(alpha),0],
        [0,0,1]
    ])
    R = np.matmul(A,B)
    #print(R)
    #print(np.linalg.det(R))
    abc = np.array([a,b,c])
    res = np.matmul(np.linalg.inv(R),abc.T)
    #print(res)
    na = res[0]
    nb = res[1]
    nc = res[2]
    return na,nb,nc

def tran(a,b,c):
    alpha = math.radians(36.795)
    beta = math.radians(78.169)
    r = 300.4
    A = np.array([
        [math.sin(beta), 0, -math.cos(beta)],
        [0, 1, 0],
        [math.cos(beta), 0, math.sin(beta)]
    ])
    B = np.array([
        [math.cos(alpha), math.sin(alpha), 0],
        [-math.sin(alpha), math.cos(alpha), 0],
        [0, 0, 1]
    ])
    R = np.matmul(A, B)
    abc = np.array([a, b, c])
    res = np.matmul(R, abc.T)
    # print(res)
    na = res[0]
    nb = res[1]
    nc = res[2]
    return na, nb, nc

if __name__ == "__main__":
    #读取
    main_point = pd.read_csv("1.csv",header=None)
    drive = pd.read_csv("2.csv",header=None)
    face = pd.read_csv("3.csv",header=None)

    main_point = np.array(main_point)
    drive = np.array(drive)
    face = np.array(face)
    point_name = np.array(main_point[:,0])

    r = 300.4

    print(main_point)
    #####转换坐标#####
    new_point = np.zeros([len(main_point),3])
    for i in range(len(main_point)):
        new_point[i,0],new_point[i,1],new_point[i,2] = tran(main_point[i,1],main_point[i,2],main_point[i,3])

    Saveexcel(new_point,"问题二转换后坐标.xlsx")

    p1,p2,p3 = retran(0,0,-300.4)

    print("新抛物面顶点坐标：",p1,p2,p3)


    #####绘制旋转节点#####

    fig1 = plt.figure()
    ax1 = plt.axes(projection='3d')

    ax1.set_zlim(-350, 0)
    for i in range(len(new_point)):
        x = new_point[i, 0]
        y = new_point[i, 1]
        z = new_point[i, 2]

        ax1.scatter3D(x, y, z, c='blue', s=2)
    #plt.show()


    #####筛选坐标#####
    point2 = pd.read_excel("问题二转换后坐标.xlsx",header=None)
    point2 = np.array(point2)

    # 寻找下方点
    low_z = r ** 2 - 150 ** 2
    low_z = math.sqrt(low_z)
    #print("最低高度为：", low_z)
    need_point = []

    for i in range(len(main_point)):
        if point2[i, 3] < -low_z:
            need_point.append([main_point[i, 0], main_point[i, 1], main_point[i, 2], main_point[i, 3]])
    print("找到的点数：", len(need_point))
    need_point = np.array(need_point)


    # 绘制主节点
    fig2 = plt.figure()
    ax2 = plt.axes(projection='3d')

    ax2.set_zlim(-350, 0)
    for i in range(len(main_point)):
        x = main_point[i, 1]
        y = main_point[i, 2]
        z = main_point[i, 3]
        color = 'blue'
        num = main_point[i, 0]
        for j in range(len(need_point)):
            if num == need_point[j,0]:
                color = 'red'

        ax2.scatter3D(x, y, z, c=color, s=2)
    #plt.show()
    
    plt.savefig("问题二1.png")
    plt.close()
    plt.savefig("问题二2.png")
    plt.close()


    #####保存需求的点#####
    data_point = []
    data = np.zeros([len(need_point),3])
    val = 0
    for i in range(len(main_point)):
        if point2[i, 3] < -low_z:
            data_point.append([main_point[val, 0]])
            data[val,0] = new_point[i,0]
            data[val,1] = new_point[i,1]
            data[val,2] = new_point[i,2]
            val = val + 1
    data_point = np.array(data_point)
    data = np.array(data)
    print(len(data_point))
    Saveexcel(data_point,'问题二找到的点.xlsx')
    Saveexcel(data,'need2.xlsx')


    #####计算t和d#####

    dz = -0.187
    t = np.zeros([len(data), 1])
    d = np.zeros([len(data), 1])
    sum = 0
    for i in range(len(data)):
        x1 = data[i,0]
        y1 = data[i,1]
        z1 = data[i,2]
        t[i, 0] = (280.774*z1 + 0.5*((675160*x1**2) + (675160*y1**2) + (315340*z1**2))**(1/2))/(x1**2 + y1**2)

        xj = x1 * t[i, 0]
        yj = y1 * t[i, 0]
        zj = z1 * t[i, 0]
        d[i] = math.sqrt((x1 - xj) ** 2 + (y1 - yj) ** 2 + (z1 - zj) ** 2)*abs(z1 - zj)/(z1 - zj)
    Saveexcel(d,"dis.xlsx")

    #####得到l#####
    d = pd.read_excel('dis.xlsx',header=None)
    d = np.array(d)
    l = np.zeros([len(d),1])
    for i in range(len(d)):
        if d[i,1]>=0:
            if d[i,1] - 0.6 > 0:
                l[i,0] = -0.6
            else:
                l[i,0] = d[i,1]
        if d[i,1]<0:
            if d[i,1] + 0.6 < 0:
                l[i,0] = 0.6
            else:
                l[i,0] = d[i,1]
    #print(l)
    Saveexcel(l,"伸长量.xlsx")

    #####求移动坐标#####
    locate = np.zeros([len(need_point),3])
    for i in range(len(data)):
        x1 = data[i, 0]
        y1 = data[i, 1]
        z1 = data[i, 2]
        x2 = l[i,0] * x1 / math.sqrt(x1**2 + y1**2 + z1**2) + x1
        y2 = l[i,0] * y1 / math.sqrt(x1**2 + y1**2 + z1**2) + y1
        z2 = l[i,0] * z1 / math.sqrt(x1**2 + y1**2 + z1**2) + z1
        locate[i,0],locate[i,1],locate[i,2] = retran(x2,y2,z2)
    Saveexcel(locate,"调整后坐标.xlsx")
