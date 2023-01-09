from collections import deque
def printinfo(arg1,start,last):
    '''*->list **->dict
    "打印任何传入的参数"'''
    print("输出: ")
    print(start,last)


# 调用printinfo 函数
printinfo(70,[1,1],[2,2])

#deque
queue = deque(["QQ","MU"])
queue.append("SKA")
queue.popleft()
print(queue)