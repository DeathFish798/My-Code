import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
    双目标帕累托分级
'''

class Utils:
    def is_dominate(obj_a, obj_b, num_obj, ):
        if type(obj_a) is not np.ndarray:
            obj_a, obj_b = np.array(obj_a), np.array(obj_b)
        res = np.array([np.sign(k) for k in obj_a - obj_b])
        res_ngt0, res_eqf1 = np.argwhere(res <= 0), np.argwhere(res == -1)
        if res_ngt0.shape[0] == num_obj and res_eqf1.shape[0] > 0:
            return True
        return False


class Pareto:
    def __init__(self, pop_size, pop_obj, ):
        self.pop_size = pop_size
        self.pop_obj = pop_obj
        self.num_obj = pop_obj.shape[1]
        self.f = []
        self.sp = [[] for _ in range(pop_size)]
        self.np = np.zeros([pop_size, 1], dtype=int)
        self.rank = np.zeros([pop_size, 1], dtype=int)
        self.cd = np.zeros([pop_size, 1])

    def __index(self, i):
        return np.delete(range(self.pop_size), i)

    def __is_dominate(self, i, j):
        return Utils.is_dominate(self.pop_obj[i], self.pop_obj[j], self.num_obj)

    def __f1_dominate(self):
        f1 = []
        for i in range(self.pop_size):
            for j in self.__index(i):
                if self.__is_dominate(i, j):
                    if j not in self.sp[i]:
                        self.sp[i].append(j)
                elif self.__is_dominate(j, i):
                    self.np[i] += 1
            if self.np[i] == 0:
                self.rank[i] = 1
                f1.append(i)
        return f1

    def fast_non_dominate_sort(self):
        rank = 1
        f1 = self.__f1_dominate()
        while f1:
            self.f.append(f1)
            q = []
            for i in f1:
                for j in self.sp[i]:
                    self.np[j] -= 1
                    if self.np[j] == 0:
                        self.rank[j] = rank + 1
                        q.append(j)
            rank += 1
            f1 = q

    def sort_obj_by(self, f=None, j=0):
        if f is not None:
            index = np.argsort(self.pop_obj[f, j])
        else:
            index = np.argsort(self.pop_obj[:, j])
        return index

    def crowd_distance(self):
        for f in self.f:
            len_f1 = len(f) - 1
            for j in range(self.num_obj):
                index = self.sort_obj_by(f, j)
                sorted_obj = self.pop_obj[f][index]
                obj_range_fj = sorted_obj[-1, j] - sorted_obj[0, j]
                self.cd[f[index[0]]] = np.inf
                self.cd[f[index[-1]]] = np.inf
                for i in f:
                    k = np.argwhere(np.array(f)[index] == i)[:, 0][0]
                    if 0 < index[k] < len_f1:
                        self.cd[i] += (sorted_obj[index[k] + 1, j] - sorted_obj[index[k] - 1, j]) / obj_range_fj


class SelectPareto:
    def __init__(self, scale_fix, scale, f, rank, cd):
        self.scale_fix = scale_fix
        self.scale = scale
        self.f = f
        self.rank = rank
        self.cd = cd
        self.num_max_front = int(0.8 * scale_fix)
        self.n_max_rank = max(self.rank[:, 0])

    def elite_strategy(self):
        ret = []
        len_f0 = len(self.f[0])
        if len_f0 > self.num_max_front and self.n_max_rank > 1:
            index_rank = np.argwhere(self.rank[:, 0] == 1)[:, 0]
            index_cd = np.argsort(-self.cd[index_rank, 0])
            ret.extend(index_rank[index_cd[:self.num_max_front]])
            for i in range(self.num_max_front, self.scale_fix):
                j = np.random.randint(1, self.n_max_rank)
                ret.extend([np.random.choice(self.f[j], 1, replace=False)[0]])
        else:
            rank = 0
            num = 0
            while True:
                num += len(self.f[rank])
                if num >= self.scale_fix:
                    break
                ret.extend(self.f[rank])
                rank += 1
            while True:
                n_more = self.scale_fix - len(ret)
                if n_more > 0:
                    index_rank = np.argwhere(self.rank[:, 0] == rank + 1)[:, 0]
                    index_cd = np.argsort(-self.cd[index_rank, 0])
                    ret.extend(index_rank[index_cd[:n_more]])
                    rank += 1
                else:
                    break
        return ret

    def champion(self):
        ret = []
        num_pareto_front = 0
        for i in range(self.scale_fix):
            if num_pareto_front >= self.num_max_front and self.n_max_rank > 1:
                j = np.random.randint(1, self.n_max_rank)
                c = np.random.choice(self.f[j], 1, replace=False)[0]
            else:
                a, b = np.random.choice(self.scale, 2, replace=False)
                if self.rank[a] < self.rank[b]:
                    c = a
                elif self.rank[a] > self.rank[b]:
                    c = b
                else:
                    if self.cd[a] > self.cd[b]:
                        c = a
                    else:
                        c = b
            ret.append(c)
            if c in self.f[0]:
                num_pareto_front += 1
        if num_pareto_front == 0:
            for index, item in enumerate(self.f[0][:self.num_max_front]):
                ret[index] = item
        return ret


'''
    代码主体
    pop_obj: 目标数据集
'''
pop_obj = pd.read_excel("new1.xlsx",sheet_name=2)
pop_obj = np.array(pop_obj)
print(pop_obj)
pop_obj[:,1] = 1 - pop_obj[:,1]
pop_obj[:,2] = pop_obj[:,2]
pop_obj = pop_obj[:,1:4]

print(pop_obj)
pareto = Pareto(pop_obj.shape[0],pop_obj)
pareto.fast_non_dominate_sort()
pareto.crowd_distance()
print(pareto.f, "# f")

choice = []
for i in range(len(pareto.f)):
    for j in range(len(pareto.f[i])):
        choice.append(pareto.f[i][j])
choice.pop()
choice = pd.array(choice)
print(choice)
np.savetxt('sort2.txt',choice,"%d")

for i, f in enumerate(pareto.f):
    plt.plot(pop_obj[f, 0], pop_obj[f, 1], "o--", label="${Rank}-{%s}$" % (i + 1))
    if i>5:
        break
plt.legend()
plt.show()

