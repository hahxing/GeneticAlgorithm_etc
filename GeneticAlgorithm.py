import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

DNA_SIZE = 10
POP_SIZE = 50
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.01
N_GENERATIONS = 50
X_BOUND = [-30, 30]
# Y_BOUND = [-3, 3]
Elite = 5

def F(xs):
    # return 3 * (1 - x) ** 2 * np.exp(-(x ** 2) - (y + 1) ** 2) - 10 * (x / 5 - x ** 3 - y ** 5) * np.exp(
    #     -x ** 2 - y ** 2) - 1 / 3 ** np.exp(-(x + 1) ** 2 - y ** 2)
    e = 2.71282
    # for i in range(10):
    #     xn2 = xs[:,i] ** 2
    #     cos =
    xn2 = xs[:,:] ** 2
    xn2 = np.sum(xn2,axis=1)
    cos = np.cos(math.pi * 2 * xs)
    cos = np.sum(cos,axis=1)
    res = (-20) * np.exp((-0.2) * np.sqrt((1 / 10) * xn2)) - np.exp((1 / 10) * cos) + 20 + e
    return res


def plot_3d(ax):
    X = np.linspace(*X_BOUND, 100)
    Y = np.linspace(*Y_BOUND, 100)
    X, Y = np.meshgrid(X, Y)
    Z = F(X, Y)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm)
    ax.set_zlim(-10, 10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.pause(3)
    plt.show()


def get_fitness(pop):
    xs = translateDNA(pop)
    pred = F(xs)
    return - (pred - np.max(
        pred)) + 1e-3  # 减去最小的适应度是为了防止适应度出现负数，通过这一步fitness的范围为[0, np.max(pred)-np.min(pred)],最后在加上一个很小的数防止出现为0的适应度


def translateDNA(pop):  # pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目
    # x_pop = pop[:, 1::2]  # 奇数列表示X
    # y_pop = pop[:, ::2]  # 偶数列表示y

    xs = np.zeros((pop.shape[0], DNA_SIZE),dtype=float)
    # pop:(POP_SIZE,DNA_SIZE)*(DNA_SIZE,1) --> (POP_SIZE,1)
    # x = x_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]
    # y = y_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (Y_BOUND[1] - Y_BOUND[0]) + Y_BOUND[0]

    for i in range(10):
        j = i * 10
        xs[:,i] = pop[:,j:j+10].dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]

    # 返回200x10的数
    return xs



def crossover_and_mutation(pop, CROSSOVER_RATE=0.8):
    new_pop = []
    for father in pop:  # 遍历种群中的每一个个体，将该个体作为父亲
        child = father  # 孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
        if np.random.rand() < CROSSOVER_RATE:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
            mother = pop[np.random.randint(POP_SIZE)]  # 再种群中选择另一个个体，并将该个体作为母亲
            cross_points = np.random.randint(low=0, high=DNA_SIZE * 10)  # 随机产生交叉的点
            child[cross_points:] = mother[cross_points:]  # 孩子得到位于交叉点后的母亲的基因
        mutation(child)  # 每个后代有一定的机率发生变异
        new_pop.append(child)

    return new_pop


def mutation(child, MUTATION_RATE=0.003):
    if np.random.rand() < MUTATION_RATE:  # 以MUTATION_RATE的概率进行变异
        mutate_point = np.random.randint(0, DNA_SIZE * 10)  # 随机产生一个实数，代表要变异基因的位置
        child[mutate_point] = child[mutate_point] ^ 1  # 将变异点的二进制为反转


def select(pop, fitness):  # nature selection wrt pop's fitness

    idx = np.random.choice(np.arange(pop.shape[0]), size=pop.shape[0], replace=True,
                           p=(fitness) / (fitness.sum()))


    return pop[idx]


def print_info(pop):
    fitness = get_fitness(pop)
    # max_fitness_index = np.argmax(fitness)
    # print("max_fitness:", fitness[max_fitness_index])
    # x, y = translateDNA(pop)
    # print("最优的基因型：", pop[max_fitness_index])
    # print("(x, y):", (x[max_fitness_index], y[max_fitness_index]))


    min_fitness_index = np.argmin(fitness)
    print("min_fitness:", fitness[min_fitness_index])
    xs= translateDNA(pop)
    print("最优的基因型：", pop[min_fitness_index])
    print("x1到x10分别为:",xs[min_fitness_index])

def test1():
    e = 2.71282
    res1 = (-20) * np.exp((-0.2) * np.sqrt((1 / 10) * 10)) - np.exp((1 / 10) * 10) + 20 + e
    print("res1:",res1)
    xs = np.ones((POP_SIZE, DNA_SIZE),dtype=float)
    res2 = F(xs)
    print("res2",res2)

def test2():
    np_arr = np.random.randint(1,100,size=(10,1))
    res = np_arr[np.argpartition(np_arr, 5,axis=0)[:5]]
    index = np.where(np_arr == res)[1]
    elites = np_arr[index,:]
    print(np_arr.shape)
    print(elites.shape)
    np_arr = np.vstack((elites, np_arr))
    print(np_arr)

if __name__ == "__main__":
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # plt.ion()  # 将画图模式改为交互模式，程序遇到plt.show不会暂停，而是继续执行
    # plot_3d(ax)

    # test1()
    # test2()
    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE * 10))  # matrix (POP_SIZE, DNA_SIZE)
    min_every_pop = []
    for _ in range(N_GENERATIONS):  # 迭代N代
        xs = translateDNA(pop)
        #print(xs)
        # if 'sca' in locals():
        #     sca.remove()
        # sca = ax.scatter(x, y, F(x, y), c='black', marker='o');
        # plt.show();
        # plt.pause(0.1)
        pop = np.array(crossover_and_mutation(pop, CROSSOVER_RATE))
        # F_values = F(translateDNA(pop)[0], translateDNA(pop)[1])#x, y --> Z matrix
        fitness = get_fitness(pop)
        pred = F(xs)
        pred1 = pred[:,np.newaxis]
        elite = pred1[np.argpartition(pred1 ,Elite,axis=0)[:Elite]]
        elite_index = np.where(pred1 == elite)[1]
        elites = pop[elite_index,:]
        min_every_pop.append(np.min(pred))
        pop = select(pop, fitness)  # 选择生成新的种群
        pop = np.vstack((elites, pop))
    print_info(pop)
    x = np.arange(1, N_GENERATIONS+1, 1)
    plt.plot(x,min_every_pop)
    plt.xlabel("种群代数")
    plt.ylabel("每代最小值")
    plt.show()
    # plt.ioff()
    # plot_3d(ax)
