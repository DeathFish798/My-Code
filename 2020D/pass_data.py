import numpy as np
import pandas as pd
from math import *
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import networkx as nx

def Saveexcel(data,filename):
    '''
        save excel
    '''
    data = pd.DataFrame(data)
    writer = pd.ExcelWriter(filename)
    data.to_excel(writer, "page_1",float_format="%.2f")
    writer.save()

def distant(x1,x2,y1,y2):
    dx = (x1 - x2)**2
    dy = (y1 - y2)**2
    dis = (dx + dy)**0.5
    return dis

def E_j(data):
    m,n = data.shape
    E = np.zeros([m,n])
    for i in range(m):
        for j in range(n):
            if data[i,j] == 0:
                e_ij = 0
        else:
            P_ij = data[i,j] / data.sum(axis = 0)[j]
            e_ij = (-1 /np.log(m)) * P_ij * np.log(P_ij)
        E[i,j] = e_ij
    res = E.sum(axis = 0)
    return res

def cul_G(point,edge):
    G = nx.DiGraph()
    for i in range(len(point)):
        G.add_node(point[i])
    for i in range(len(edge)):
        G.add_edge(edge[i,0],edge[i,1],weight = float(edge[i,2]))
    dc = nx.degree_centrality(G)
    cluster = nx.clustering(G)
    return dc,cluster

if __name__ == "__main__":
    fullevent = pd.read_csv("fullevents.csv")
    passevent = pd.read_csv("passingevents.csv")
    passevent = np.array(passevent)
    fullevent = np.array(fullevent)
    huskies_pass = []
    for i in range(len(passevent)):
        if passevent[i,1] == "Huskies":
            data = []
            for j in range(len(passevent[i])):
                data.append(passevent[i,j])
        huskies_pass.append(data)
    huskies_pass = np.array(huskies_pass)
    print(huskies_pass)

    #####find point locate#####
    player = set()
    for i in range(len(huskies_pass)):
        player.add(huskies_pass[i,2])
        player.add(huskies_pass[i,3])
    player = list(player)
    print("num of players in huskies:",len(player))
    point = []
    xy = np.zeros([len(player),3])
    for i in range(len(player)):
        point.append([player[i]])
    #print(point)
    for i in range(len(passevent)):
        passer = passevent[i,2]
        getter = passevent[i,3]
        for j in range(len(player)):
            if passer == player[j]:
                xy[j, 0] += passevent[i, -4]
                xy[j, 1] += passevent[i, -3]
                xy[j, 2] += 1
            if getter == player[j]:
                xy[j, 0] += passevent[i, -2]
                xy[j, 1] += passevent[i, -1]
                xy[j, 2] += 1
    #print(xy)
    for i in range(len(xy)):
        xy[i,0] = xy[i,0] / xy[i,2]
        xy[i,1] = xy[i,1] / xy[i,2]
        point[i].append(xy[i,0])
        point[i].append(xy[i,1])
    Saveexcel(point,"point.xlsx")
    #network graph
    G = nx.DiGraph()
    for i in range(len(point)):
        G.add_node(player[i],pos = (xy[i,0],xy[i,1]))
    nx.draw_networkx(G, with_labels=True)
    #plt.show()

    #####passway#####
    normal = 0
    special = 0
    for i in range(len(passevent)):
        if passevent[i,6] == "Simple pass":
            normal += 1
        else:
            special += 1
    print(normal,special)

    #####construct graph#####
    edge = []
    dis = []
    success = [0,0]
    fail = [0,0]
    for i in range(len(passevent)):
        passer = passevent[i, 2]
        getter = passevent[i, 3]
        a = -1
        data = []
        for j in range(len(player)):
            if passer == player[j]:
                a += 1
            if getter == player[j]:
                a += 1
        if a >= 0:
            if a == 1:
                if passevent[i, 4] == "1H":
                    success[0] += 1
                else:
                    success[1] += 1
            else:
                if passevent[i, 4] == "1H":
                    fail[0] += 1
                else:
                    fail[1] += 1
            #pass way
            if passevent[i,6] == "Simple pass":
                b = 1
            else:
                b = 4
            #distance
            l = distant(passevent[i,-4],passevent[i,-3],passevent[i,-2],passevent[i,-1])
            dis.append(l)
            #weight
            w = (b + a) * exp(l / 86)

            data.append(passer)
            data.append(getter)
            data.append(w)
            edge.append(data)
    edge = np.array(edge)
    dis = sorted(dis)
    dis = np.array(dis)
    print(success,fail)
    print(dis[int(len(dis) * 0.95)])
    print(edge)
    Saveexcel(edge,"edge.xlsx")
    Saveexcel(dis,"dis.xlsx")

    #####calculate degree#####
    passcore = np.zeros([len(player),1])
    for i in range(len(edge)):
        passer = edge[i,0]
        getter = edge[i,1]
        w = float(edge[i,2])
        for j in range(len(player)):
            if player[j] == passer:
                passcore[j,0] += w
            if player[j] == getter:
                passcore[j,0] += w
    Saveexcel(player,"player.xlsx")
    Saveexcel(passcore,"passcore.xlsx")

    #####match diversity#####
    '''
    S_data = np.array([
        [10,5,4],
        [3,10,6]
    ])
    S_data = S_data.T
    scaler1 = MinMaxScaler()
    res1 = scaler1.fit_transform(S_data)
    S_w = E_j(S_data)
    S_w = 1 - S_w
    S_w = S_w / sum(S_w)
    print("weight matrix of t:\n",S_w)

    M_data = np.zeros([2,2])
    for i in range(len(huskies_pass)):
        if huskies_pass[i,4] == "1H":
            if float(huskies_pass[i,5]) <= 1350:
                M_data[0,0] += 1
            else:
                M_data[0,1] += 1
        else:
            if float(huskies_pass[i,5]) <=1350:
                M_data[1,0] += 1
            else:
                M_data[1,1] += 1
    print("initial matrix of m:\n",M_data.T)
    scaler2 = MinMaxScaler()
    res2 = scaler2.fit_transform(M_data.T)
    M_w = E_j(M_data.T)
    M_w = 1 - M_w
    M_w = M_w / sum(M_w)
    print("weight matrix of m:",M_w)
    '''
    dc,clu = cul_G(player,edge)
    col1 = dc.keys()
    col2 = dc.values()
    col3 = clu.values()
    col1 = list(col1)
    col2 = list(col2)
    col3 = list(col3)
    data = list([col1, col2, col3])
    data = np.array(data)
    Saveexcel(data.T, "data_huskies.xlsx")


    #####all team#####
    passtimedata = []
    fouldata = []

    for i in range(19):
        team_name = "Opponent" + str(i + 1)
        team_pass = []
        for j in range(len(passevent)):
            if passevent[j, 1] == team_name:
                data = []
                for k in range(len(passevent[j])):
                    data.append(passevent[j, k])
                team_pass.append(data)
        team_pass = np.array(team_pass)

        #select player
        point = set()
        for j in range(len(team_pass)):
            point.add(team_pass[j, 2])
            point.add(team_pass[j, 3])
        point = list(point)
        print("----------")
        print(team_name,"\n",len(point))

        #construct graph
        edge = []
        dis = []
        success = [0, 0]
        fail = [0, 0]
        for j in range(len(passevent)):
            passer = passevent[j, 2]
            getter = passevent[j, 3]
            a = -1
            data = []
            for k in range(len(point)):
                if passer == point[k]:
                    a += 1
                if getter == point[k]:
                    a += 1
            if a >= 0:
                if a == 1:
                    if passevent[j, 4] == "1H":
                        success[0] += 1
                    else:
                        success[1] += 1
                else:
                    if passevent[j, 4] == "1H":
                        fail[0] += 1
                    else:
                        fail[1] += 1
                # pass way
                if passevent[j, 6] == "Simple pass":
                    b = 1
                else:
                    b = 4
                # distance
                l = distant(passevent[j, -4], passevent[j, -3], passevent[j, -2], passevent[j, -1])
                dis.append(l)
                # weight
                w = (b + a) * exp(l / 86)

                data.append(passer)
                data.append(getter)
                data.append(w)
                edge.append(data)
        edge = np.array(edge)
        print(len(edge))

        #culculate G
        dc,clu = cul_G(point,edge)

        #culculate 50 pass
        passtime = 0
        count = 50
        for j in range(len(passevent)):
            if int(passevent[j,0]) == i + 1:
                passer = passevent[j, 2]
                getter = passevent[j, 3]
                data = []
                a = -1
                for k in range(len(player)):
                    if passer == player[k]:
                        a += 1
                    if getter == player[k]:
                        a += 1
                if a >= 0:
                    if count == 50:
                        passtime = passevent[j,5]
                        print("first time",passtime)
                    count -= 1
                    if count == 0:
                        passtime = passevent[j,5] - passtime
                        break
        print("50 passtime:",passtime)

        #count foul
        foul = 0
        for j in range(len(fullevent)):
            if fullevent[j, 1] == "Opponent" + str(i + 1):
                if fullevent[j, 6] == "Foul":
                    foul += 1
        print("foul:", foul)

        passtimedata.append(passtime)
        fouldata.append(foul)
        col1 = dc.keys()
        col2 = dc.values()
        col3 = clu.values()
        col1 = list(col1)
        col2 = list(col2)
        col3 = list(col3)
        data = list([col1,col2,col3])
        data = np.array(data)
        #Saveexcel(data.T,"data_opponent{}.xlsx".format(str(i+1)))
    print("----------")
    print(len(passtimedata),len(fouldata))
    col1 = []
    for i in range(19):
        col1.append(i + 1)
    data = list([col1,passtimedata,fouldata])
    data = np.array(data).T
    Saveexcel(data,"pass_foul_data.xlsx")
