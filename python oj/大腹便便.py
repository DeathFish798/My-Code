import matplotlib.pyplot as plt
from numpy import *
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']


def F(theta,fine):
        F = abs(sin(N / 2 * (pi * cos(theta) - fine))
                / sin((pi * cos(theta) - fine) / 2)
                * sin(theta)
                )
        return F

if __name__ == "__main__":
        N = 7
        theta = arange(0 * pi, 2 * pi, 0.01 * pi)
        fai = [0,pi / 2,pi * 2 / 3]
        res = []
        leg = []
        for i in range(len(fai)):
                fig, ax = plt.subplots()
                ax = plt.subplot(111, projection='polar')
                res.append(F(theta,fai[i]))
                plt.plot(theta,res[i])
                title = "phi = {:.3f}".format(fai[i])
                leg.append(title)
                plt.title(title)
                plt.show()
        fig, ax = plt.subplots()
        ax = plt.subplot(111, projection='polar')
        theta = arange(0 * pi, 2 * pi, 0.01 * pi)
        f = abs(sin(theta))
        plt.plot(theta, res[0],theta,res[1],'r',theta,res[2],'g',theta,f)
        leg.append("单独电偶极子天线")
        plt.legend(leg)
        plt.show()