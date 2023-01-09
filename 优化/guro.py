import numpy as np
import gurobipy as gb
from gurobipy import GRB

#  maximize
#        x +   y + 2 z
#  subject to
#        x + 2 y + 3 z <= 4
#        x +   y       >= 1
#        x, y, z binary

mod = gb.Model("mip1")

x={}
#var
x = mod.addVar(vtype=GRB.BINARY,name="x")
y = mod.addVar(vtype=GRB.BINARY,name="y")
z = mod.addVar(vtype=GRB.BINARY,name="z")

#obj
mod.setObjective(x + y + 2*z,GRB.MAXIMIZE)
mod.status == gb.GRB.Status.OPTIMAL

#bound
mod.addConstr(x + 2*y + 3*z <=4,"c0")
mod.addConstr(x + y >=1,"c1")

#model
mod.optimize()

#print
for v in mod.getVars():
    print('%s %g' % (v.varName,v.x))

print('Obj:%g' % mod.objVal)

