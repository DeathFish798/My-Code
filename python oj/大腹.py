from numpy import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    fig,ax = plt.subplots()
    ax = plt.subplot(111,projection='polar')
    theta = arange(0 * pi, 2 * pi, 0.01 * pi)
    f = abs(sin(theta))
    plt.plot(theta,f)
    plt.show()