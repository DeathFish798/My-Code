import numpy as np
def floyd(gragh):
    '''
    floyd求最短路径
    有向图，权为正，两点间最短路
    '''
    n = len(gragh)
    path = []   #创建中转站矩阵
    for i in range(n):
        a = []
        for j in range(n):
            a.append(-1)
        path.append(a)

    for i in range(n):
        for u in range(n):
            for v in range(n):
                if u!=v:
                    if (gragh[u][v]>gragh[u][i]+gragh[i][v]):
                        gragh[u][v] = gragh[u][i]+gragh[i][v]
                        path[u][v] = i
    return gragh,path

def search(start,end,gragh,path):
    '''
    搜索两点之间的距离及道路
    :param start: 起点点
    :param end: 终点
    '''
    u = start
    v = end
    print("两点距离为：",gragh[u][v],"  路径为：",start,"->",end=" ")
    while(path[u][v]!=-1):
        u = path[u][v]
        print(u,end=" -> ")
    print(v)

if __name__ == "__main__":
    inf = 999
    gragh = np.array([
        [0,5,inf,7],
        [inf,0,4,2],
        [3,3,0,2],
        [inf,inf,1,0]
    ])
    new_gragh,path = floyd(gragh)
    print(new_gragh)
    print(path)
    search(0,2,new_gragh,path)