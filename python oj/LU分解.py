#线性代数进阶大作业
#LU分解
import numpy as np

class LU_Decomposition():
    """
        metrix:待分解的矩阵
        method:选择的方法(默认为auto,可选择:"Doolettle","Crout","Cholesky")
        b:线性方程组 Ax = b 中的矩阵b
    """
    def __init__(self,metrix,b = [],method = "auto"):
        self.X = metrix
        self.method = method
        self.row = metrix.shape[0]
        self.col = metrix.shape[1]
        self.b = b
        self.flag = 1

    def examine(self):
        """
            判断是否可以进行LU分解
        """
        if self.row != self.col:
            print("矩阵不是方阵")
            return 0
        else:
            #判断n阶主子式是否大于零
            for i in range(self.row):
                B = []

                for j in range(i+1):
                    list = []
                    for k in range(i+1):
                        list.append(self.X[j,k])
                    B.append(list)

                if np.linalg.det(B) < 0:
                    print("{}阶主子式不大于等于零".format(str(i+1)))
                    return 0
            return 1

    def decomposation(self):
        if self.examine() == 0:
            print("矩阵不符合分解条件")
            return 0,0
        else:
            if self.method == "Doolittle":
                L,U = self.doolittle()
                return L,U

            if self.method == "Crout":
                L, U = self.crout()
                return L, U

            if self.method == "Cholesky":
                if not (self.X.T == self.X).all():
                    print("矩阵非对称正定矩阵，无法进行Cholesky分解")
                    self.flag = 0
                    return 0,0
                else:
                    L,U = self.cholesky()
                    return L,U

            if self.method == "auto":
                if (self.X.T == self.X).all():
                    L, U = self.cholesky()
                    return L, U
                else:
                    L, U = self.doolittle()
                    return L, U

    def doolittle(self):
        #用doolittle方法分解
        U = np.zeros([self.row, self.col])
        L = np.eye(self.row)  # L对角线为1

        U[0, :] = self.X[0, :]
        for i in range(self.row):
            L[i, 0] = self.X[i, 0] / U[0, 0]

        for i in range(1, self.row):
            for j in range(i, self.row):
                U[i, j] = self.X[i, j] - np.dot(L[i, :i], U[:i, j])
                if j + 1 < self.row:
                    L[j + 1, i] = (self.X[j + 1, i] - np.dot(L[j + 1, :i], U[:i, i])) / U[i, i]

        print("---分解完成---")
        print("L:\n{}\n".format(L))
        print("U:\n{}\n------------".format(U))
        self.L = L
        self.U = U
        return L, U

    def crout(self):
        #用crout方法分解
        L = np.zeros([self.row, self.col])
        U = np.eye(self.row)  # L对角线为1

        L[:, 0] = self.X[:, 0]
        for i in range(self.row):
            U[0, i] = self.X[0, i] / L[0, 0]

        for i in range(1, self.row):
            for j in range(i, self.row):
                L[j, i] = self.X[j, i] - np.dot(U[:i, i], L[j, :i])
                if j + 1 < self.row:
                    U[i, j + 1] = (self.X[i, j + 1] - np.dot(U[:i, j + 1], L[i, :i])) / L[i, i]

        print("---分解完成---")
        print("L:\n{}\n".format(L))
        print("U:\n{}\n------------".format(U))
        self.L = L
        self.U = U
        return L, U

    def cholesky(self):
        L = np.zeros([self.row,self.col])
        for i in range(self.row):
            L[i,i] = (self.X[i,i] - np.dot(L[i,:i],L[i,:i].T))**0.5
            for j in range(i+1,self.row):
                L[j,i] = (self.X[j,i] - np.dot(L[j,:i],L[i,:i].T))/L[i,i]
        self.L = L
        self.U = L.T
        print("---分解完成---")
        print("L:\n{}\n".format(L))
        print("U:\n{}\n------------".format(L.T))
        return L, L.T

    def solver(self):
        """
            求解线性方程组 Ax = b
        """
        if self.col != self.b.shape[0]:
            print("b不符合条件")
            return 0,0

        elif self.examine() == 0:
            return 0

        elif self.flag == 0:
            return 0

        else:
            b = self.b
            y = np.dot(np.linalg.inv(self.L),b)
            x = np.dot(np.linalg.inv(self.U),y)
            print("\n---求解方程---")
            print("x的结果为:")
            print(x)
            print("----------")
            return x


if __name__ == "__main__":
    """
        输入部分
        LU分解: A = LU
        线性方程求解: Ax = b
    """
    A = np.array([
        [5,-2,0],
        [-2,3,-1],
        [0,-1,1]
    ])
    b = np.array([
        [1],
        [3],
        [2]
    ])

    #运行
    Dec = LU_Decomposition(A,b)
    L,U = Dec.decomposation() #LU分解
    x = Dec.solver() #求解线性方程组
