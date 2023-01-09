#        x +   y + 2 z
#  subject to
#        x + 2 y + 3 z <= 4
#        x +   y       >= 1
#        x, y, z binary

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp



#创建模型
m = gp.Model("matrix1")

# 将矩阵变量添加到模型中：在这种情况下，矩阵变量由 3 个二进制变量的一维数组组成。
x = m.addMVar(3, vtype=GRB.BINARY, name="x")

# 目标是通过使用重载@ 运算符计算常数向量和矩阵变量之间的点积来构建的。请注意，常数向量必须与我们的矩阵变量具有相同的长度。
# 第二个参数表明意义是最大化。
obj = np.array([1.0, 1.0, 2.0])
m.setObjective(obj @ x, GRB.MAXIMIZE)

# 构建（稀疏）约束矩阵
# 注意，我们将大于约束乘以将其乘以-1<转换为小于约束。
val = np.array([1.0, 2.0, 3.0, -1.0, -1.0]) # 两个约束的左侧系数值
row = np.array([0, 0, 0, 1, 1]) # 分别对应行。（0表示第一行，1表示第二行）
col = np.array([0, 1, 2, 0, 1]) # 分别表示的列。（0表示第一列）

A = sp.csr_matrix((val, (row, col)), shape=(2, 3))

# 构建 rhs 向量。也就是约束右边的值。
rhs = np.array([4.0, -1.0])

# 添加约束模型。使用重载的@运算符构建一个线性矩阵表达式，然后使用重载的小于或等于运算符添加两个约束
m.addConstr(A @ x <= rhs, name="c")

# 优化模型
m.optimize()

v = x.X
print(v)
print('Obj: %g' % m.objVal)

