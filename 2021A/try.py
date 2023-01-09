import numpy as np

if __name__ == '__main__':
    x1, y1 = 0, 50
    x3, y3 = 100, -10
    x2, y2 = 100, 200
    sample_size = 1000000
    theta = np.arange(0, 1, 0.001)
    x = theta * x1 + (1 - theta) * x2
    y = theta * y1 + (1 - theta) * y2

    x = theta * x1 + (1 - theta) * x3
    y = theta * y1 + (1 - theta) * y3

    x = theta * x2 + (1 - theta) * x3
    y = theta * y2 + (1 - theta) * y3

    rnd1 = np.random.random(size=sample_size)
    rnd2 = np.random.random(size=sample_size)
    rnd2 = np.sqrt(rnd2)
    x = rnd2 * (rnd1 * x1 + (1 - rnd1) * x2) + (1 - rnd2) * x3
    y = rnd2 * (rnd1 * y1 + (1 - rnd1) * y2) + (1 - rnd2) * y3

    print(x,y)
    print(len(x),len(y))

