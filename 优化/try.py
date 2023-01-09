import numpy as np
from gurobipy import *
from gurobipy import GRB

c = [3,10,7,6,11,9]
a = [
    [12,9,25,20,17,13],
    [35,42,18,31,56,49],
    [37,53,28,24,29,20]
]
b = [60,150,125]

#创建模型
model = Model("rururu")

#创建变量
x = model.addVars(6,vtype=GRB.CONTINUOUS,lb=0,ub=1,name='x')
model.update()

#创建目标
model.setObjective(x.prod(c),GRB.MINIMIZE)

#创建约束
for i in range(3):
    model.addConstr(x.prod(a[i])>=b[i])

#执行优化
model.optimize()
for v in model.getVars():
    print(v.varName,v.x)