import numpy as np
import matplotlib.pyplot as plt
import cma

def rosenbrock(self, x):
    fitness = 0
    fitness += x[0]**10 + x[1]**10
    return fitness

es = cma.CMAEvolutionStrategy(2 * [0],  0.5)
es.optimize(cma.ff.rosen)
es.result_pretty()

