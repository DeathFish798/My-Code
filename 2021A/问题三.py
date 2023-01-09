import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fsolve

def Saveexcel(data,filename):
    data = pd.DataFrame(data)
    writer = pd.ExcelWriter(filename)
    data.to_excel(writer, "page_1", float_format='%.3f')
    writer.save()

def mente(x1,x2,x3,y1,y2,y3):
    sample_size = 1000
    theta = np.arange(0, 1, 0.001)
    x = theta * x1 + (1 - theta) * x2
    y = theta * y1 + (1 - theta) * y2

    x = theta * x1 + (1 - theta) * x3
    y = theta * y1 + (1 - theta) * y3

    x = theta * x2 + (1 - theta) * x3
    y = theta * y2 + (1 - theta) * y3

    rnd1 = np.random.random(size=sample_size)
    rnd2 = np.random.random(size=sample_size)
    rnd2 = np.sqrt(rnd2)
    x = rnd2 * (rnd1 * x1 + (1 - rnd1) * x2) + (1 - rnd2) * x3
    y = rnd2 * (rnd1 * y1 + (1 - rnd1) * y2) + (1 - rnd2) * y3
    return x,y

def findz(xg,yg):
    global bx,by,bz
    r = 300.4
    return bz - (- bx**2 + 2*bx*xg - by**2 + 2*by*yg + r**2 - xg**2 - yg**2)**(1/2)

def func(paramlist):
    global x1, x2, x3, y1, y2, y3, z1, z2, z3
    x,y,z=paramlist[0],paramlist[1],paramlist[2]
    r = 300.4
    return [(x-x1)**2+(y-y1)**2+(z-z1)**2-r**2,(x-x2)**2+(y-y2)**2+(z-z2)**2-r**2,(x-x3)**2+(y-y3)**2+(z-z3)**2-r**2]

def findball(func):
    s = fsolve(func,[0,0,0])
    return s[0],s[1],s[2]

def judge1(bx,by,bz,xg,yg,zg):
    X = bx - xg
    Y = by - yg
    Z = bz - zg
    x = (1602*X*Z + 5*X**2*xg + 5*Y**2*xg - 5*Z**2*xg + 10*X*Z*zg)/(5*(X**2 + Y**2 + Z**2))
    y = (1602*Y*Z + 5*X**2*yg + 5*Y**2*yg - 5*Z**2*yg + 10*Y*Z*zg)/(5*(X**2 + Y**2 + Z**2))
    d = math.sqrt(x**2 + y**2)
    if d<=0.5*2/2:
        return 1
    else:
        return 0

def judge2(bx,by,bz,xg,yg,zg):
    X = bx - xg
    Y = by - yg
    Z = bz - zg
    x = (1602*X*Z + 5*X**2*xg + 5*Y**2*xg - 5*Z**2*xg + 10*X*Z*zg)/(5*(X**2 + Y**2 + Z**2))
    y = (1602*Y*Z + 5*X**2*yg + 5*Y**2*yg - 5*Z**2*yg + 10*Y*Z*zg)/(5*(X**2 + Y**2 + Z**2))
    d = math.sqrt(x**2 + y**2)
    if d<=0.5:
        return 1
    else:
        return 0


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
    # 读取
    main_point = pd.read_csv("1.csv", header=None)
    drive = pd.read_csv("2.csv", header=None)
    face = pd.read_csv("3.csv", header=None)


    main_point = np.array(main_point)
    drive = np.array(drive)
    face = np.array(face)
    point_name = np.array(main_point[:, 0])

    r = 300.4
    low_z = r ** 2 - 150 ** 2
    low_z = math.sqrt(low_z)

    need_point = pd.read_excel("need2.xlsx")#需求点基准坐标
    need_point = np.array(need_point)
    print(need_point)

    locate = pd.read_excel("调整后坐标.xlsx")#需求点调整后坐标
    locate = np.array(locate)

    for i in range(len(locate)):
        locate[i,0],locate[i,1],locate[i,2] = tran(locate[i,0],locate[i,1],locate[i,2])


    #寻找连接点
    set = {'apple'}
    set.pop()
    for i in need_point:
        set.add(i[0])

    for i in set.copy():
        for j in range(len(face)):
            if i == face[j,0]:
                set.add(face[j,1])
                set.add(face[j,2])
    #print(set)

    # 返回需求点
    new_need = np.zeros([len(set), 3])
    need_name = []
    val = 0
    for i in set:
        for j in range(len(main_point)):
            if main_point[j, 0] == str(i):
                need_name.append(main_point[j, 0])
                new_need[val, 0] = main_point[j, 1]
                new_need[val, 1] = main_point[j, 2]
                new_need[val, 2] = main_point[j, 3]
                break
        val = val + 1
    need_name = np.array(need_name)

    print(need_name)
    print(need_point)
    #转化需求点坐标
    for i in range(len(new_need)):
        new_need[i,0],new_need[i,1],new_need[i,2] = tran(new_need[i,0],new_need[i,1],new_need[i,2])
    new_locate = new_need.copy()
    for i in range(len(new_need)):
        for j in range(len(locate)):
            if need_point[j,0] == need_name[i]:
                new_locate[i,0] = locate[j,0]
                new_locate[i,1] = locate[j,1]
                new_locate[i,2] = locate[j,2]

    print("new_need:",new_need)
    print()
    print("new_locate:",new_locate)
    #new_need为所有点基准点坐标，new_locate为所有点转换后的坐标
    #Saveexcel(new_need,"newneed.xlsx")
    #Saveexcel(new_locate,"newloc.xlsx")

    #####求出所有板#####
    ban = []
    for i in range(len(need_point)):
        num = need_point[i,0]
        for j in range(len(face)):
            if num == face[j,0]:
                ban.append([face[j,0],face[j,1],face[j,2]])
    ban = np.array(ban)
    print("共找到{}块板".format(len(ban)))

    #####蒙特卡洛#####

    #移动后
    all_num = 0
    success = 0
    for i in range(len(ban)):
        p1 = ban[i,0]
        p2 = ban[i,1]
        p3 = ban[i,2]
        for j in range(len(new_locate)):
            if p1 == need_name[j]:
                x1 = new_locate[j,0]
                y1 = new_locate[j,1]
                z1 = new_locate[j,2]
            if p2 == need_name[j]:
                x2 = new_locate[j,0]
                y2 = new_locate[j,1]
                z2 = new_locate[j,2]
            if p3 == need_name[j]:
                x3 = new_locate[j,0]
                y3 = new_locate[j,1]
                z3 = new_locate[j,2]
        bx,by,bz = findball(func)
        xg,yg = mente(x1,x2,x3,y1,y2,y3)
        xg = np.array(xg)
        yg = np.array(yg)
        all_num = all_num + len(xg)
        for u in range(len(xg)):
            zg = findz(xg[u],yg[u])
            if zg>low_z:
                all_num = all_num - 1
            else:
                success = success + judge1(bx,by,bz,xg[u],yg[u],zg)
    print(all_num)
    print(success)
    print("旋转后馈源舱的接收比：",success / all_num)

    #移动前
    all_num = 0
    success = 0
    for i in range(len(ban)):
        p1 = ban[i, 0]
        p2 = ban[i, 1]
        p3 = ban[i, 2]
        for j in range(len(new_locate)):
            if p1 == need_name[j]:
                x1 = new_need[j, 0]
                y1 = new_need[j, 1]
                z1 = new_need[j, 2]
            if p2 == need_name[j]:
                x2 = new_need[j, 0]
                y2 = new_need[j, 1]
                z2 = new_need[j, 2]
            if p3 == need_name[j]:
                x3 = new_need[j, 0]
                y3 = new_need[j, 1]
                z3 = new_need[j, 2]
        bx, by, bz = findball(func)
        xg, yg = mente(x1, x2, x3, y1, y2, y3)
        xg = np.array(xg)
        yg = np.array(yg)
        all_num = all_num + len(xg)
        for u in range(len(xg)):
            zg = findz(xg[u], yg[u])
            if zg > low_z:
                all_num = all_num - 1
            else:
                success = success + judge2(bx, by, bz, xg[u], yg[u], zg)
    print(all_num)
    print(success)
    print("旋转前馈源舱的接收比：", success / all_num)

