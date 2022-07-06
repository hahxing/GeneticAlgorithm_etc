import numpy as np
import random as rand
import time
import matplotlib.pyplot as plt


class ArtificialBeeColony:

    def __init__(self, D, Lb, Ub, n, generation, ans, func):
        self.fitnessRes = []
        self.D = D
        self.Lb = Lb
        self.Ub = Ub
        self.n = n
        self.generation = generation
        self.ans = ans
        self.func = func
        self.bestFunc = 0
        self.bestFoodSourceArray = np.zeros(D)
        self.foodSourceArray = np.ones((n, D)) #解集
        self.tmpFoodSourceArray = np.ones(D)
        self.funcArray = np.ones(n)
        self.fitnessArray = np.ones(n)
        self.trialArray = np.zeros(n)
        self.pArray = np.zeros(n)

    # 适应度
    def fitness(self, X):
        if (X >= 0):
            return 1 / (1 + X)
        else:
            return 1 + abs(X)

    # 侦察蜂
    def generateNew(self, X, Xp):
        Xnew = X + rand.uniform(-1, 1) * (X - Xp)
        if (Xnew < self.Lb):
            return self.Lb
        elif (Xnew > self.Ub):
            return self.Ub
        else:
            return Xnew

    # 对每一个解变异并更新矩阵
    def updateSolution(self, index):
        randVariableToChange = rand.randint(0, self.D - 1)
        randPartner = index
        while (randPartner == index):
            randPartner = rand.randint(0, self.n - 1)

        for j in range(self.D):
            self.tmpFoodSourceArray[j] = self.foodSourceArray[index][j]
        self.tmpFoodSourceArray[randVariableToChange] = self.generateNew(
            self.foodSourceArray[index][randVariableToChange], self.foodSourceArray[randPartner][randVariableToChange])

        if (self.ans == 0):
            oriVal = self.funcArray[index]
            newVal = self.func(self.tmpFoodSourceArray, self.D)
        elif (self.ans == 1):
            oriVal = self.fitnessArray[index]
            newVal = self.fitness(self.func(self.tmpFoodSourceArray, self.D))
        # 如果新解优于原先解，trial则为0，否则+1
        if (newVal < oriVal):
            self.foodSourceArray[index][randVariableToChange] = self.tmpFoodSourceArray[randVariableToChange]
            self.funcArray[index] = self.func(self.tmpFoodSourceArray, self.D)
            self.fitnessArray[index] = self.fitness(self.func(self.tmpFoodSourceArray, self.D))
            self.trialArray[index] = 0 #第 i 个解的实验次数
        else:
            self.trialArray[index] = self.trialArray[index] + 1

    def printLocalBestSolution_MAX(self):
        localBest = int(np.where(self.funcArray == self.funcArray.max())[0])
        print("Local Best Food Source:", self.foodSourceArray[localBest])
        print("local Best F(x) =", self.funcArray.max())
        if (self.funcArray.max() > self.bestFunc):
            for i in range(self.D):
                self.bestFoodSourceArray[i] = self.foodSourceArray[localBest][i]
            self.bestFunc = self.funcArray.max()

    def printLocalBestSolution_MIN(self):
        min = self.funcArray.min()
        localBest = int(np.where(self.funcArray == min)[0])
        self.fitnessRes.append(min)
        print("Local Best Food Source:", self.foodSourceArray[localBest])
        print("local Best F(x) =", self.funcArray.min())
        if (self.funcArray.min() < self.bestFunc):
            for i in range(self.D):
                self.bestFoodSourceArray[i] = self.foodSourceArray[localBest][i]
            self.bestFunc = self.funcArray.min()

    def printCurrentSolution(self):
        print("==================================")
        print("foodSourceArray\n", self.foodSourceArray)
        print("funcArray\n", self.funcArray)
        print("fitnessArray\n", self.fitnessArray)
        print("trialArray\n", self.trialArray)
        print("==================================")

    def init(self):
        if (self.ans == 0):
            self.bestFunc = float("inf")
        elif (self.ans == 1):
            self.bestFunc = -float("inf")

        for i in range(self.n):
            for j in range(self.D):
                self.foodSourceArray[i][j] = rand.uniform(self.Lb, self.Ub) #(n, D)
            self.funcArray[i] = self.func(self.foodSourceArray[i, :], self.D) #(n)
            self.fitnessArray[i] = self.fitness(self.funcArray[i]) #(n)

    def doRun(self):
        start = time.time()
        self.init()

        for gen in range(self.generation):
            print("Generation:", gen + 1)

            # Employed Bee Phase
            for i in range(self.n):
                self.updateSolution(i)

            # 跟随蜂阶段,解被选取概率
            for i in range(self.n):
                self.pArray[i] = self.fitnessArray[i] / self.fitnessArray.sum()
            # 引领蜂邻域搜索
            for i in range(self.n):
                if (rand.random() < self.pArray[i]):
                    self.updateSolution(i)

            if (self.ans == 0):
                self.printLocalBestSolution_MIN()
            elif (self.ans == 1):
                self.printLocalBestSolution_MAX()

            # Scout Bee Phase
            limit = 1
            # 更新funcArray等,并重新侦察
            for i in range(self.n):
                if (self.trialArray[i] > limit):
                    for j in range(self.D):
                        self.foodSourceArray[i][j] = rand.uniform(self.Lb, self.Ub) #(n, D)
                    self.funcArray[i] = self.func(self.foodSourceArray[i, :], self.D) #(n)
                    self.fitnessArray[i] = self.fitness(self.funcArray[i]) #(n)
                    self.trialArray[i] = 0
        end = time.time()
        print("============================================")
        print("执行时间：%f 秒" % (end - start))
        print("Best Food Source:", self.bestFoodSourceArray)
        print("Best F(x) =", self.bestFunc)

def test(X,D):
    x1 = X[0]
    x2 = X[1]
    return x1**10 + x2**10

def RastriginFunc(X, D):
    funsum = 0
    for i in range(D):
        x = X[i]
        funsum += x**2-10*np.cos(2*np.pi*x)+10
    return funsum


Gen = 50
# abc = ArtificialBeeColony(30, -5.12, 5.12, n=100, generation=Gen,ans= 0,func= RastriginFunc)
abc = ArtificialBeeColony(2, -5.12, 5.12, 10, 50, 0, test)
abc.doRun()
plt.figure()
plt.title("ABC")
plt.xlabel("generation", size=14)
plt.ylabel("fitness", size=14)
t = [t for t in range(1, Gen + 1)]
plt.plot(t, abc.fitnessRes, color='b', linewidth=2)
plt.show()
