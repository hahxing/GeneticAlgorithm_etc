import numpy as np
import matplotlib.pyplot as plt
import datetime

# 种群规模
N = 50
# x1:[-0.3,12.1]
# x2:[4.1-5,8]
bound_x1 = [-5.12,5.12]
bound_x2 = [-5.12,5.12]
# 初始化x1、x2
# 计算sigma 初始化种群
# 初始化可以修改为先生成[2,N]的正态分布 再给每一个计算具体的值
X = np.random.random([2, N])
X[0] = (bound_x1[1] - bound_x1[0]) * X[0] + bound_x1[0]
X[1] = (bound_x2[1] - bound_x2[0]) * X[1] + bound_x2[0]
# 迭代的代数 作为程序终止的条件
GEN = 50
# 变异强度 标准正态分布
# 期望最大值
E_MAX = 0.001
# x1、x2
res_x_y = []
# sigma
sigma = np.random.normal([1, 2])
# 最大的适应度值
MIN_FITNESS = float('inf')


# 计算适应度的函数
def F(x1, x2):
    # return 21.5 + x1 * np.sin(4 * np.pi * x1) + x2 * np.sin(20 * np.pi * x2)
    Z = x1 ** 10 + x2 ** 10
    return Z

# 生成的mu个儿子
son = np.zeros([2, N])

# 当前代
step = 1

if __name__ == '__main__':
    start_time = datetime.datetime.now()
    print("开始时间:",start_time)
    fitness = []
    for _ in range(GEN):
        # 计算sigma 生成子代
        num = 0
        while num < N:
            # 当有多个参数时 整一个while循环 遍历所有的sigma元素
            # 暂时两个
            if sigma[0] < 0:
                sigma[0] = 0.001
            else:
                sigma[0] += np.sqrt(sigma[0]) * np.random.normal(loc=0, scale=1)
            if sigma[1] < 0:
                sigma[1] = 0.001
            else:
                sigma[1] += np.sqrt(sigma[1]) * np.random.normal(loc=0, scale=1)
            X0 = X[:, num] + np.random.normal(loc=0, scale=1) * sigma

            # 判断是否越界
            if (bound_x1[0] <= X0[0] <= bound_x1[1]) and (bound_x2[0] <= X0[1] <= bound_x2[1]):
                son[:, num] = X0
                num += 1
        # 适应度计算评价以及选择后代
        # 使用q随机竞争法 q=0.9N
        theta = 0.98
        q = theta * N
        # 得分
        w = np.zeros(2 * N)

        # 合并son和X
        temp = np.ndarray([2, 2 * N])
        for i in range(2):
            temp[i, :N] = son[i]
            temp[i, N:2 * N] = X[i]
        # 循环体
        for i in range(2 * N):
            p = 0
            # 随机找到q个 对当前的第i个个体进行适应度值比较 计算得分w[i]
            # 每次都找随机的q个个体作为判断样本
            while p < q:
                # 在2N个中间随机找到一个
                j = np.random.randint(0, 2 * N)
                # i、j不相等 做不做判断？
                p += 1
                # 计算当前个体和随机找到的j个体的适应度值
                i_fitness = F(temp[:, i][0], temp[:, i][1])
                j_fitness = F(temp[:, j][0], temp[:, j][1])
                # i获胜
                if (i_fitness <= j_fitness):
                    w[i] += 1
        # 进行排序 从小到大排序索引
        arg = w.argsort()
        # 获取N个较大的个体的索引
        # 并重新生成X
        X = temp[:, arg[N: 2 * N]]
        # step += 1
        # 最大适应值对应的x、y
        res_x_y = X[:, -1]
        MIN_FITNESS_TEMP = F(res_x_y[0], res_x_y[1])
        fitness.append(MIN_FITNESS_TEMP)
        # 更新最大值
        # if MIN_FITNESS > MIN_FITNESS_TEMP:
        #     MIN_FITNESS = MIN_FITNESS_TEMP
        #     # 期望的输出 达到值后退出循环
        #     if MIN_FITNESS_TEMP < E_MAX:
        #         end_time = datetime.datetime.now()
        #         print("结束时间：",end_time)
        #         print("花费时间：",end_time-start_time)
        #         break
        #     print(MIN_FITNESS)
        step += 1
    print("结果：")
    print("step:" + str(step))
    print("x：" + str(res_x_y[0]))
    print("y：" + str(res_x_y[1]))
    # print(MIN_FITNESS)
    plt.figure()
    plt.title("EP")
    plt.xlabel("generation", size=14)
    plt.ylabel("fitness", size=14)
    t = [t for t in range(1, GEN + 1)]
    plt.plot(t, fitness, color='b', linewidth=2)
    plt.show()
