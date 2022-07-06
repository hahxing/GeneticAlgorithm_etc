# Population-Based Incremental Learning
import matplotlib.pyplot as plt
import random
import numpy as np

def get_num(proba):
    if random.random() < proba:
        return 1
    return 0

def translateDNA(X):
    xs = np.zeros(n_x,dtype=float)
    for i in range(n_x):
        j = i * DNA_SIZE
        xs[i] = X[j:j+DNA_SIZE].dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]
    return xs

def F(X):
    res = 0
    for i in range(0,len(X)):
        res += X[i] ** 2 - 10 * np.cos(np.pi * 2 * X[i]) + 10
    return res
def F1(X):
    res = 0
    res = X[0] ** 2 + X[1] ** 2
    return res
def optimize(learn_rate, neg_learn_rate, pop_size, num_best_vec_to_update_from, num_worst_vec_to_update_from, vec_len,
             optimisation_cycles, eval_f, eps=0.01, vec_storage=None):
    # vector initialisation(150)
    vec = np.full(vec_len, 0.5, dtype=float)
    # initialise population(50,150)
    population = np.empty((pop_size, vec_len), dtype=int)
    scores = [None for _ in range(pop_size)]
    # initialise best result
    best_of_all = [float("inf"), None]
    if vec_storage is not None:
        vec_storage.append(list(vec))
    for i in range(optimisation_cycles):
        # solution vectors generation
        for j in range(pop_size):
            for k in range(vec_len):
                population[j][k] = get_num(vec[k])
            # vector evaluation
            xs = translateDNA(population[j])
            scores[j] = eval_f(xs)
        # 根据适应值函数的得分，对种群进行从大到小的排序
        sorted_res = sorted(zip(scores, population), key=lambda x:x[0], reverse=False)
        best = sorted_res[:num_best_vec_to_update_from]
        worst = sorted_res[-num_worst_vec_to_update_from:]
        fitness.append(best[0][0])
        # update best_of_all
        if best_of_all[0] > best[0][0]:
            best_of_all = (best[0][0], list(best[0][1]))
        # update vector
        for v in best:
            vec += 2 * learn_rate * (v[1] - 0.5)
        for v in worst:
            vec -= 2 * neg_learn_rate * (v[1] - 0.5)
        # vector correction if elements outside [0, 1] range
        for j in range(vec_len):
            if vec[j] < 0:
                vec[j] = 0 + eps
            elif vec[j] > 1:
                vec[j] = 1 - eps
        # store vec?
        if vec_storage is not None:
            vec_storage.append(list(vec))
    return best_of_all[0],best_of_all[1]

if __name__ == '__main__':
    n_x = 2
    DNA_SIZE = 10
    X_BOUND = [-5.12, 5.12]
    Gen = 50
    l = []
    fitness = []
    min, X= optimize(learn_rate=0.02,neg_learn_rate= 0.02, pop_size=100,num_best_vec_to_update_from= 2,num_worst_vec_to_update_from= 2,
                   vec_len=n_x * DNA_SIZE,optimisation_cycles= Gen,eval_f= F1, vec_storage=l)
    X = np.array(X)

    print(min)
    print(translateDNA(X))
    plt.figure()
    plt.title("PBIL")
    plt.xlabel("generation", size=14)
    plt.ylabel("fitness", size=14)
    t = [t for t in range(1, Gen + 1)]
    plt.plot(t, fitness, color='b', linewidth=2)
    plt.show()