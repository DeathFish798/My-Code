import copy
import random
import matplotlib.pyplot as plt

# 无向图
sample = [
    [1, 378, 78],
    [2, 327, 119],
    [3, 232, 266],
    [4, 314, 311],
    [5, 255, 477],
    [6, 296, 513],
    [7, 510, 438],
    [8, 628, 246],
]

vertex_count = len(sample)  # 顶点数


def draw1(ls1, ls2):
    plt.plot(ls1, ls2)
    plt.savefig("pic1.png")
    plt.close()

def draw(ls):
    x = []
    y = []
    for i in ls:
        x.append(sample[i - 1][1])
        y.append(sample[i - 1][2])
    #####调整坐标范围#####
    plt.xlim(200, 700)
    plt.ylim(0, 600)
    plt.plot(x, y, marker='o', mec='r', mfc='w')
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel(u"X")
    plt.ylabel("Y")
    plt.title("PATH")
    plt.savefig("pic2.png")

def city_dist(sample, city1, city2):
    if city1 == city2:
        return 9999
    loc1 = []
    loc2 = []
    findtag = 0
    dist = 0
    for i in range(len(sample)):
        if city1 == sample[i][0]:
            loc1.append(sample[i][1])
            loc1.append(sample[i][2])
            findtag += 1
        if city2 == sample[i][0]:
            loc2.append(sample[i][1])
            loc2.append(sample[i][2])
            findtag += 1
        if findtag == 2:
            break
    dist = sqrt(((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2))
    return dist


def evalute(sample, individual):
    distance = 0
    for i in range(len(individual) - 1):  # vertex_count = len(individual)-1
        distance += city_dist(sample, individual[i], individual[i + 1])
    return distance


def find_pheromone(pheromone, city1, city2):
    # 信息素表示 [1,2,a] 即城市1与2之间的信息素为a
    for i in range(len(pheromone)):
        if city1 in pheromone[i] and city2 in pheromone[i]:
            return pheromone[i][2]
    print("return 0")
    return 0


def update_pehromone(pheromone, city1, city2, new_pehromone):  # 更新信息素
    for i in range(len(pheromone)):
        if city1 in pheromone[i] and city2 in pheromone[i]:
            pheromone[i][2] = 0.9 * pheromone[i][2] + new_pehromone


def get_sum(cur_city, pheromone, allowed_city, alpha, beta):
    sum = 0
    for i in range(len(allowed_city)):
        sum += (find_pheromone(pheromone, cur_city, allowed_city[i]) ** alpha) * (
                    city_dist(sample, cur_city, allowed_city[i]) ** beta)
    return sum


def AntColony_Algorithm(sample):
    # 重要参数
    ant_count = 20  # 蚁群个体数
    alpha = 5
    beta = 1
    loop_count = 50  # 迭代次数

    ant_colony = []
    ant_individual = []
    allowed_city = []  # 待选城市
    city = []
    p = []  # 记录选择某一城市概率
    pheromone = []
    draw_ls1 = []
    draw_ls2 = []
    best_dist = 9999
    best_route = []

    for i in range(1, vertex_count + 1):  # 初始化信息素
        for j in range(i + 1, vertex_count + 1):
            pheromone.append([i, j, 100])  # 信息素初始化不能为0
    for i in range(1, vertex_count + 1):  # 初始化城市和概率
        city.append(i)

    # 进行迭代
    for i in range(loop_count):
        if i % 10 == 0:
            print(i)
            draw(best_route)
        draw_ls1.append(i)
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
            for m in range(vertex_count):  # 确定了起始城市，循环次数-1
                cur_city = ant_colony[singal_ant][-1]
                # 单个蚂蚁遍历所有城市以确定下一城市
                for j in range(len(allowed_city)):
                    probability = ((find_pheromone(pheromone, cur_city, allowed_city[j]) ** alpha) * (
                                city_dist(sample, cur_city, allowed_city[j]) ** beta)) / get_sum(cur_city, pheromone,
                                                                                                 allowed_city, alpha,
                                                                                                 beta)
                    p.append(probability)
                    # 求累积概率
                cumulative_probability = [0]
                for j in range(len(p)):
                    cumulative_probability.append(cumulative_probability[j] + p[j])
                # 自然选择  轮盘赌概率选择下一城市
                temp_random = random.random()  # 产生(0,1)随机数
                for j in range(1, len(cumulative_probability)):
                    if temp_random > cumulative_probability[j - 1] and temp_random < cumulative_probability[j]:
                        ant_colony[singal_ant].append(allowed_city[j - 1])  # 在单个蚂蚁中添加被选择的下一城市
                        del allowed_city[j - 1]
                        break
                p.clear()
            ant_colony[singal_ant].append(ant_colony[singal_ant][0])
            # 计算每只蚂蚁的路径长度并更新所有蚂蚁路径上的信息素
        for j in range(ant_count):
            if evalute(sample, ant_colony[j]) < best_dist:
                best_dist = evalute(sample, ant_colony[j])
                best_route = copy.deepcopy(ant_colony[j])
            for k in range(len(ant_colony[j]) - 1):
                update_pehromone(pheromone, ant_colony[j][k], ant_colony[j][k + 1],
                                 10000 / city_dist(sample, ant_colony[j][k], ant_colony[j][k + 1]))
        draw_ls2.append(best_dist)
        ant_colony.clear()

    print("outcome")
    print(pheromone)
    print(best_route)
    print(best_dist)
    draw(best_route)
    draw1(draw_ls1, draw_ls2)


AntColony_Algorithm(sample)
