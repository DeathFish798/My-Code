import sys
def prim(graph,n):
    '''
    prim算法求最小生成树
    解决问题：无向图，最小生成树
    '''
    lowcost = []
    start = []
    cost = 0
    for i in range(n):
        lowcost.append(graph[0][i])
        start.append(0)
    now = 0
    lowcost[0] = -1 #用“-1"来标记被选择的点，
    for i in range(n-1):
        min = sys.maxsize
        for j in range(n):
            if(min > lowcost[j] and lowcost[j] != -1):
                min = lowcost[j]
                now = j
        cost += min
        print(traver(start[now]),"->",traver(now),": ",min)
        lowcost[now] = -1

        for k in range(n): #更新lowcost
            if(lowcost[k] > graph[now][k]):
                lowcost[k] = graph[now][k]
                start[k] = now

    return cost

def traver(x):
    '''
    把数字转换为字母
    '''
    return chr(x+97)


if __name__=='__main__':
    inf = sys.maxsize
    '''
    输入临接矩阵，其中无法连接为inf
    '''
    graph = [[inf,6,1,5,inf,inf],
             [6,inf,5,inf,3,inf],
             [1,5,inf,5,6,4],
             [5,inf,5,inf,inf,2],
             [inf,3,6,inf,inf,6],
             [inf,inf,4,2,6,inf]]
    print(prim(graph,len(graph)))