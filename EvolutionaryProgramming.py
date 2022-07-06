import numpy as np
import matplotlib.pyplot as plt
import datetime
import math
"""
进化规划-ep 实现多峰值函数的最大值计算
"""
# 种群规模
N = 50
x_num = 2
bound = [-5.12, 5.12]
X = np.random.uniform(-5.12,5.12,[N, x_num])
for i in range(x_num):
    X[i] = (bound[1] - bound[0]) * X[i] + bound[0]
GEN = 50
res = []
sigma = np.random.normal(0,1,size=x_num)
MIN_FITNESS = float('inf')

def Rastrigr(Xs):
    # res = 0
    # for x in X:
    #     res += x ** 2 - 10 * np.cos(math.pi * 2 * x) + 10
    # return res
    X, Y = Xs
    Z = X ** 10 + Y ** 10
    return Z
son = np.zeros([N,x_num])

step = 0

if __name__ == '__main__':
    start_time = datetime.datetime.now()
    print("开始时间:",start_time)
    fitness = []
    for gen in range(GEN):
    #     pefect = float('inf')
    #     pefect_index = 0
    #     for index in range(N):
    #         temp = Rastrigr(X[index])
    #         if temp < pefect:
    #             pefect = temp
    #             pefect_index = index
        num = 0
        while num < N:
            for i in range(x_num):
                if sigma[i] < 0:
                    sigma[i] = 0.05
                else:
                    sigma[i] += np.sqrt(sigma[i]) * np.random.normal(loc=0, scale=1)
            X0 = X[num] + np.random.normal(loc=0, scale=1) * sigma
            # flag = 1
            # for i in range(x_num):
            #     if(bound[0] > X0[i] or X0[i] > bound[1]):
            #         flag = 0
            #         print("******重新变异******")
            # if flag:
            #     son[num] = X0
            #     num += 1
            for i in range(x_num):
                if(bound[0] > X0[i] or X0[i] > bound[1]):
                    X0[i] = np.random.uniform(-5.12,5.12)
                    # X0[i] = 0
                    # X0 = X[pefect_index]
                    # break
            son[num] = X0
            num += 1
        # 适应度计算评价以及选择后代
        # 使用q随机竞争法 q=0.9N
        theta = 0.90
        q = theta * N
        # 得分
        w = np.zeros(2 * N)
        temp = np.ndarray([ 2 * N, x_num])
        temp[0:N,:] = son
        temp[N:2 * N, :] = X
        # 循环体
        for i in range(2 * N):
            p = 0
            while p < q:
                # 在2N个中间随机找到一个
                j = np.random.randint(0, 2 * N)
                p += 1
                i_fitness = Rastrigr(temp[i, :])
                j_fitness = Rastrigr(temp[j, :])
                if (i_fitness < j_fitness):
                    w[i] += 1
        # 进行排序 从小到大排序索引
        arg = w.argsort()
        # 获取N个较大的个体的索引
        # 并重新生成X
        X = temp[arg[N: 2 * N],:]
        res = X[-1]
        MAX_FITNESS_TEMP = Rastrigr(res)
        fitness.append(MAX_FITNESS_TEMP)
        print("第"+str(gen+1)+"代：",MAX_FITNESS_TEMP)
        step += 1
    print("结果：")
    print("step:" + str(step))
    print("x：" + str(res))
    # print(MIN_FITNESS)
    plt.figure()
    plt.title("EP")
    plt.xlabel("generation", size=14)
    plt.ylabel("fitness", size=14)
    t = [t for t in range(1, GEN + 1)]
    plt.plot(t, fitness, color='b', linewidth=2)
    plt.show()

