import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

Xi = np.array([160, 165, 158, 172, 159, 176, 160, 162, 171])
Yi = np.array([58, 63, 57, 65, 62, 66, 58, 59, 62])

def func(p, x):
    k, b = p
    return k*x+b

def error(p, x, y):
    return func(p, x)-y

# k,b的初始值，可以任意设定,经过几次试验，发现p0的值会影响cost的值：Para[1]
p0 = [80, 20]

# 把error函数中除了p0以外的参数打包到args中(使用要求)
Para = leastsq(error, p0, args=(Xi, Yi))

# 读取结果
k, b = Para[0]
print("k=", k, "b=", b)
print(Para[0])
def S(k,b):
    error=np.zeros(k.shape)
    for x,y in zip(Xi,Yi):
        error+=(y-(k*x+b))**2
    return error
S(k,b)

# 画样本点
plt.figure(figsize=(8, 6))  # 指定图像比例： 8：6
plt.scatter(Xi, Yi, color="green", label="YB", linewidth=2)

# 画拟合直线
x = np.linspace(150, 190, 100)  # 在150-190直接画100个连续点
y = k*x+b  # 函数式
plt.plot(x, y, color="red", label="NH", linewidth=2)
plt.legend()  # 绘制图例
