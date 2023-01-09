import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

excel=pd.read_excel("electric.xlsx")
yonghu=np.array(excel)
print(yonghu)
print(yonghu[0])
print(yonghu[:,1])
new=np.sort(yonghu,1)
print(new)

x=yonghu[:,0]
y1=yonghu[:,1]
y2=100*yonghu[:,2]

#plt.plot(x,y1,x,y2)
#plt.show()

A=np.array([[3,5],
           [2,7]])
print(A)
b=np.array([[7],
           [3]])
print(b)
x=np.linalg.solve(A,b)
print(x)

print(np.linalg.inv(A))