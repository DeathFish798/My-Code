import gurobipy

# 创建模型
EMPLOYEE, MIN, MAX, COST, START, END = gurobipy.multidict({
	"SMITH"   : [6, 8, 30, 6, 20], "JOHNSON": [6, 8, 50, 0, 24], 'WILLIAMS': [6, 8, 30, 0, 24],
	'JONES'   : [6, 8, 30, 0, 24], 'BROWN': [6, 8, 40, 0, 24], 'DAVIS': [6, 8, 50, 0, 24],
	'MILLER'  : [6, 8, 45, 6, 18], 'WILSON': [6, 8, 30, 0, 24], 'MOORE': [6, 8, 35, 0, 24],
	'TAYLOR'  : [6, 8, 40, 0, 24], 'ANDERSON': [2, 3, 60, 0, 6], 'THOMAS': [2, 4, 40, 0, 24],
	'JACKSON' : [2, 4, 60, 8, 16], 'WHITE': [2, 6, 55, 0, 24], 'HARRIS': [2, 6, 45, 0, 24],
	'MARTIN'  : [2, 3, 40, 0, 24], 'THOMPSON': [2, 5, 50, 12, 24], 'GARCIA': [2, 4, 50, 0, 24],
	'MARTINEZ': [2, 4, 40, 0, 24], 'ROBINSON': [2, 5, 50, 0, 24]})
REQUIRED = [1, 1, 2, 3, 6, 6, 7, 8, 9, 8, 8, 8, 7, 6, 6, 5, 5, 4, 4, 3, 2, 2, 2, 2]
MODEL = gurobipy.Model("Work Schedule")

x = {}
# 创建变量
x = MODEL.addVars(EMPLOYEE, range(24), range(1, 25), vtype=gurobipy.GRB.BINARY)

# 更新变量环境
MODEL.update()

# 创建目标函数
MODEL.setObjective(gurobipy.quicksum((j - i) * x[d, i, j] * COST[d] for d in EMPLOYEE for i in range(24) for j in range(i + 1, 25)), sense=gurobipy.GRB.MINIMIZE)

# 创建约束条件
MODEL.addConstrs(x.sum(d) <= (gurobipy.quicksum(x[d, i, j] for i in range(START[d], END[d] + 1) for j in range(i + 1, END[d] + 1) if MIN[d] <= j - i <= MAX[d])) for d in EMPLOYEE)
MODEL.addConstrs(gurobipy.quicksum(x[d, i, j] for i in range(START[d], END[d] + 1) for j in range(i + 1,END[d] + 1) if MIN[d] <= j - i <= MAX[d]) <= 1 for d in EMPLOYEE)
MODEL.addConstrs(gurobipy.quicksum(x[d, i, j] for d in EMPLOYEE for i in range(24) for j in range(i + 1, 25) if i <= c < j) >= REQUIRED[c] for c in range(24))

# 执行最优化
MODEL.optimize()

'''
# 输出结果
x_axis = list(range(24))
y_axis = []
data = []

if MODEL.status == gurobipy.GRB.Status.OPTIMAL:
	solution = [k for k, v in MODEL.getAttr('x', x).items() if v == 1]
	for d, i, j in solution:
		print(f"The working time of {d} is from {i} to {j}")
		y_axis.append(d)
		data.extend([[time, d, COST[d]] for time in range(i, j)])
	for c in range(24):
		member = [d for d, i, j in solution if i <= c < j]
		print(f'The member of staff from {c} -{c+1}: {",".join(member)}')
'''

for v in x:
    print(v)

print('Obj:%g' % MODEL.objVal)