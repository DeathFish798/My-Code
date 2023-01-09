def Dijkstra(gragh, begin, end):
    '''
    :param gragh:邻接矩阵
    :param begin:出发点
    :param end:目的点

    解决问题：有向图，正权重，单源最短路
    '''
    Dis = gragh[begin - 1]
    print('初始化:', end='')
    print(Dis)
    path = [0] * len(Dis)   #存储到这个节点前最后一个经过的节点
    # 需要依次更新表
    for i in range(1, len(Dis)):
        a = [(key, value) for key, value in enumerate(Dis)]
        mid = sorted(a, key=lambda x: x[1])[i]  #确定第i个点
        mid_index = mid[0]
        mid_value = mid[1]

        print('确定第{}个点:'.format(mid_index+1), end='')
        print(Dis)
        # 更新Dis表
        for k in range(len(Dis)):
            if mid_value + gragh[mid_index][k] < Dis[k]:
                Dis[k] = mid_value + gragh[mid_index][k]
                path[k] = mid_index + 1     #更新路径，保存更新的中间节点
        # 如果最小节点找到了目的节点，可以直接退出
        if mid_index + 1 == end:
            break
    value = end
    node_list = []
    for _ in range(len(Dis)):
        value = path[value - 1]
        if value != 0:
            node_list.append(value)
        else:
            break
    node_list.append(begin)
    node_list.reverse()
    node_list.append(end)
    node_list = list(map(str, node_list))
    print('最短径长为{0},最短路径为:'.format(Dis[end-1]), end='')
    print('->'.join(node_list))

if __name__ == "__main__":
    inf = 999
    '''
        输入临接矩阵，其中无法连接为inf
    '''
    gragh = [
        [0,11,5,inf,11,inf,19,inf,30,inf,48,inf],
        [inf,0,inf,11,5,inf,11,inf,19,inf,30,inf],
        [inf,inf,0,11,inf,inf,inf,inf,inf,inf,inf,inf],
        [inf,inf,inf,0,inf,12,5,inf,11,inf,19,inf],
        [inf,inf,inf,inf,0,12,inf,inf,inf,inf,inf,inf],
        [inf,inf,inf,inf,inf,0,inf,12,5,inf,11,inf],
        [inf,inf,inf,inf,inf,inf,0,12,inf,inf,inf,inf],
        [inf,inf,inf,inf,inf,inf,inf,0,inf,13,5,inf],
        [inf,inf,inf,inf,inf,inf,inf,inf,0,13,inf,inf],
        [inf,inf,inf,inf,inf,inf,inf,inf,inf,0,inf,0],
        [inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,0],
        [inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,0]
    ]
    Dijkstra(gragh, 1, 12)