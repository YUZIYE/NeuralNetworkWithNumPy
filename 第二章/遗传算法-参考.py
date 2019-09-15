import numpy
import ga

import numpy as np 

def f(x):
    x1, x2 = x[:, 0], x[:, 1]
    u = 2.5 * x1
    v = 2.5 * x2
    return u * u + 5 * v * np.sin(v) + 5 * v

def selection(pop, fit, num):
    """
    选择一部分优秀个体
    pop:所有个体
    fit:健康程度
    num:选择个数
    输出:选择的个体
    """
    idx = np.argsort(fit)
    return pop[idx[:num]]

def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    for k in range(offspring_size[0]):
        parent1_idx = k%parents.shape[0]
        parent2_idx = (k+1)%parents.shape[0]
        offspring[k, 0:1] = parents[parent1_idx, 0:1]
        offspring[k, 1:] = parents[parent2_idx, 1:]
    return offspring

def mutation(offspring_crossover, num_mutations=1):
    mutations_counter = numpy.uint8(offspring_crossover.shape[1] / num_mutations)
    for idx in range(offspring_crossover.shape[0]):
        gene_idx = mutations_counter - 1
        for mutation_num in range(num_mutations):
            # The random value to be added to the gene.
            random_value = numpy.random.uniform(-0.5, 0.5, 1)
            offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] + random_value
            gene_idx = gene_idx + mutations_counter
    return offspring_crossover

"""
The y=target is to maximize this equation ASAP:
    y = w1x1+w2x2+w3x3+w4x4+w5x5+6wx6
    where (x1,x2,x3,x4,x5,x6)=(4,-2,3.5,5,-11,-4.7)
    What are the best values for the 6 weights w1 to w6?
    We are going to use the genetic algorithm for the best possible values after a number of generations.
"""

# Inputs of the equation.
equation_inputs = [4,-2,3.5,5,-11,-4.7]

# Number of the weights we are looking to optimize.
num_weights = len(equation_inputs)

"""
Genetic algorithm parameters:
    Mating pool size
    Population size
"""

new_population = numpy.random.uniform(low=-3.0, high=3.0, size=[20, 2])
print(new_population)

best_outputs = []
outputs = []
num_generations = 20
for itr in range(num_generations):
    # 计算群体健康程度
    fitness = f(new_population)
    best_outputs.append(new_population[np.argmax(fitness)])
    # 选择函数较小的部分样本
    parents = selection(new_population, fitness, 10)
    # 交叉重组
    offspring_crossover = crossover(parents, [10, 2])

    # Adding some variations to the offspring using mutation.
    offspring_mutation = mutation(offspring_crossover, num_mutations=2)

    # Creating the new population based on the parents and offspring.
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation
    
fitness = f(new_population)
# Then return the index of that solution corresponding to the best fitness.
best_match_idx = numpy.where(fitness == numpy.max(fitness))

print("Best solution : ", new_population[best_match_idx, :])
print("Best solution fitness : ", fitness[best_match_idx])


import matplotlib.pyplot
matplotlib.pyplot.plot(best_outputs)
matplotlib.pyplot.xlabel("Iteration")
matplotlib.pyplot.ylabel("Fitness")
matplotlib.pyplot.show()