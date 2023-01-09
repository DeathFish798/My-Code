import copy
import random
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family']=['Heiti TC']    #显示中文
plt.rcParams['axes.unicode_minus']=False

#顶点位置
arc = [
    [1, 378, 78],
    [2, 327, 119],
    [3, 232, 266],
    [4, 314, 311],
    [5, 255, 477],
    [6, 296, 513],
    [7, 510, 438],
    [8, 628, 246],
]
city_num = len(arc)  # 城市数
city_name = ["南宁","河内","曼谷","金边","吉隆坡","新加坡","文莱","马尼拉"]

def draw_iter(iter, res):
    '''
        画出迭代图
    '''
    plt.plot(iter, res)
    plt.title(u"迭代曲线")
    plt.xlabel(u"迭代次数")
    plt.ylabel(u"最优解")
    plt.savefig("pic1.png")
    plt.close()

def draw_loc(route):
    '''
        画出坐标图
    '''
    x = []
    y = []
    global city_name
    for i in route:
        x.append(arc[i - 1][1])
        y.append(arc[i - 1][2])
    #####调整坐标范围#####
    plt.xlim(200, 700)
    plt.ylim(0, 600)
    plt.plot(x, y, marker='o', mec='r', mfc='w')
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel(u"X")
    plt.ylabel("Y")
    plt.title(u"旅行路线图")
    for i in range(city_num):
        plt.annotate(city_name[i],xy=(arc[i][1],arc[i][2]))
    plt.savefig("pic2.png")
    plt.close()

def city_dist(arc, city1, city2):
    '''
        返回城市距离
    '''
    if city1 == city2:
        return 9999
    loc1 = []
    loc2 = []
    findtag = 0
    dist = 0
    for i in range(len(arc)):
        if city1 == arc[i][0]:
            loc1.append(arc[i][1])
            loc1.append(arc[i][2])
            findtag += 1
        if city2 == arc[i][0]:
            loc2.append(arc[i][1])
            loc2.append(arc[i][2])
            findtag += 1
        if findtag == 2:
            break
    dist = np.sqrt(((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2))
    return dist

def evalute(arc, individual):
    '''
        进化条件
    '''
    distance = 0
    for i in range(len(individual) - 1):
        distance += city_dist(arc, individual[i], individual[i + 1])
    return distance

def find_pheromone(pheromone, city1, city2):
    '''
        计算信息素矩阵
    '''
    for i in range(len(pheromone)):
        if city1 in pheromone[i] and city2 in pheromone[i]:
            return pheromone[i][2]
    return 0


def update_pehrom(pherom, city1, city2, new_pehromone):
    """
        更新信息素
    """
    for i in range(len(pherom)):
        if city1 in pherom[i] and city2 in pherom[i]:
            pherom[i][2] = 0.9 * pherom[i][2] + new_pehromone


def get_sum(cur_city, pherom, allowed_city, alpha, beta):
    sum = 0
    for i in range(len(allowed_city)):
        sum += (find_pheromone(pherom, cur_city, allowed_city[i]) ** alpha) * (
                    city_dist(arc, cur_city, allowed_city[i]) ** beta)
    return sum

def AntColony_Algorithm(arc):
    """
        蚁群算法
    """
    ant_count = 20  # 蚁群个体数
    loop_count = 50  # 迭代次数
    alpha = 5
    beta = 1

    ant_colony = []
    ant_individual = []
    allowed_city = []  # 记录待选城市
    city = []
    p = []  # 记录选择城市概率
    pherom = [] # 记录信息素
    iter = []   # 记录迭代次数
    res = []    # 记录最优解
    best_dist = 9999
    best_route = []

    for i in range(1, city_num + 1):  # 初始化信息素
        for j in range(i + 1, city_num + 1):
            pherom.append([i, j, 100])  # 信息素初始化不能为0
    for i in range(1, city_num + 1):  # 初始化城市和概率
        city.append(i)

    # 进行迭代
    for i in range(loop_count):
        if i % 10 == 0:
            print("迭代次数:",i)
            draw_loc(best_route)
        iter.append(i)
        # 随机产生蚂蚁起始点
        start_city = []
        for j in range(ant_count):
            start_city.append(random.randint(1, len(city)))
        for j in range(len(start_city)):
            ant_individual.append(start_city[j])
            ant_colony.append(copy.deepcopy(ant_individual))
            ant_individual.clear()
        # 所有蚂蚁完成遍历
        for singal_ant in range(ant_count):
            # 单个蚂蚁完成路径
            allowed_city = copy.deepcopy(city)
            allowed_city.remove(ant_colony[singal_ant][0])
            for m in range(city_num):
                cur_city = ant_colony[singal_ant][-1]
                # 单个蚂蚁遍历所有城市
                for j in range(len(allowed_city)):
                    probability = ((find_pheromone(pherom, cur_city, allowed_city[j]) ** alpha) * (
                                city_dist(arc, cur_city, allowed_city[j]) ** beta)) / get_sum(cur_city, pherom,
                                            allowed_city, alpha, beta)
                    p.append(probability)
                # 求累积概率
                cul_p = [0]
                for j in range(len(p)):
                    cul_p.append(cul_p[j] + p[j])
                # 自然选择  轮盘赌概率选择下一城市
                temp = random.random()
                for j in range(1, len(cul_p)):
                    if temp > cul_p[j - 1] and temp < cul_p[j]:
                        ant_colony[singal_ant].append(allowed_city[j - 1])  # 在单个蚂蚁中添加被选择的下一城市
                        del allowed_city[j - 1]
                        break
                p.clear()
            ant_colony[singal_ant].append(ant_colony[singal_ant][0])
            # 计算每只蚂蚁的路径长度并更新所有蚂蚁路径上的信息素
        for j in range(ant_count):
            if evalute(arc, ant_colony[j]) < best_dist:
                best_dist = evalute(arc, ant_colony[j])
                best_route = copy.deepcopy(ant_colony[j])
            for k in range(len(ant_colony[j]) - 1):
                update_pehrom(pherom, ant_colony[j][k], ant_colony[j][k + 1],
                                 10000 / city_dist(arc, ant_colony[j][k], ant_colony[j][k + 1]))
        res.append(best_dist)
        ant_colony.clear()

    print("结果如下：")
    print(best_dist)
    draw_loc(best_route)
    draw_iter(iter, res)
    return best_route


if __name__ == "__main__":
    best_route = AntColony_Algorithm(arc)
    for i in best_route[:-1]:
        print(city_name[i-1],"->",end=" ")
    print(city_name[best_route[-1]-1])