import numpy as np
from collections import deque

def BFS(gragh,last):
    '''
    广度优先搜索
    '''
    count =0
    code = 0
    linkcode = 0
    step = 0
    t1,t2,t3,t4 = gragh
    dx = np.array([ [-1, 1, 0, 0],
                    [-1, -1, 0, 0],
                    [-1, 0, 1, 0],
                    [-1, 0, -1, 0],
                    [-1, 0, 0, 1],
                    [-1, 0, 0, -1],
                    [1, 1, 0, 0],
                    [1, -1, 0, 0],
                    [1, 0, 1, 0],
                    [1, 0, -1, 0],
                    [1, 0, 0, 1],
                    [1, 0, 0, -1],
                    [1, 0, 0, 0 ],
                    [-1,0, 0, 0 ]])
    node = [] #存储路径，格式为[linkcode,code,step,n1,n2,n3,n4]
    queue = deque()
    queue.append([linkcode,code,step,t1,t2,t3,t4])
    node.append([linkcode,code,step,t1,t2,t3,t4])
    count = 1
    while(queue):
        none,code,step,t1,t2,t3,t4 = queue.popleft()
        linkcode = code
        for i in range(14):
            n1,n2,n3,n4 = [t1,t2,t3,t4]+dx[i]
            if((n1==0 or n1 ==1) and (n2==0 or n2 ==1) and (n3==0 or n3 ==1) and (n4==0 or n4 ==1)):
                if(not ((n2 == n3 and n1 != n2) or (n3 == n4 and n1 != n3))):
                    node.append([linkcode,count,step+1,n1,n2,n3,n4])
                    queue.append([linkcode,count,step+1,n1,n2,n3,n4])
                    count = count + 1
            if ([n1,n2,n3,n4]==last):
                print("最小步长为:",step+1)
                print([1,1,1,1])
                for u in range(step+1):
                    linkcode,code,none,n1,n2,n3,n4=node[linkcode]
                    print("前一状态：", [linkcode, code, none, n1, n2, n3, n4])
                return 0


if __name__ =="__main__":
    gragh = np.array([0,0,0,0])
    print(gragh)
    BFS(gragh,[1,1,1,1])

