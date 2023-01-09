import numpy as np
from collections import deque

def BFS(gragh,start,last):
    '''
    广度优先搜索
    :param start:起点
    :param last:终点
    '''
    dx = [1,0,-1,0]
    dy = [0,-1,0,1]
    tx,ty = start
    gragh[tx,ty]=7
    row = np.size(gragh,0)
    col = np.size(gragh,1)
    step = 0

    nodex = np.zeros([row,col]) #储存路径
    nodey = np.zeros([row,col])

    queue = deque()
    queue.append([tx,ty,0])
    while(queue):
        tx,ty,step = queue.popleft()
        for i in range(4):
            nx = tx + dx[i]
            ny = ty + dy[i]
            if (nx>=0 and ny>=0 and nx<row and ny<col):
                if(gragh[nx,ny]==1):
                    queue.append([nx,ny,step+1])
                    gragh[nx][ny] = 2
                    nodex[nx][ny] = tx
                    nodey[nx][ny] = ty
            if ([nx,ny]==last):
                print("最小步长为:",step+1)
                lx,ly=last
                gragh[lx][ly]=8
                fx = int(nodex[nx][ny]) #回溯路径
                fy = int(nodey[nx][ny])
                for j in range(step-1):
                    gragh[fx][fy] = 3
                    fx = int(nodex[fx][fy])
                    fy = int(nodey[fx][fy])
                return gragh
    return gragh




if __name__ =="__main__":
    '''
        输入图，0表示障碍，1表示接通
        def BFS(图论,start,last):
        start:起点
        last:终点
    '''
    gragh=np.array([[0, 1, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 0, 1],
                    [0, 1, 1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 0, 0, 1],
                    [0, 1, 1, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1]])

    print(gragh)
    new_gragh = BFS(gragh,[0,1],[6,6])
    print(new_gragh)
