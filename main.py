# some inspiration taken from:
# https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=4f12bb2e8cdb0fd339b87f7c74234ea61b17d2d4
# https://www.kecl.ntt.co.jp/icl/as/members/yamada/ICGA91.PDF
# https://www.sciencedirect.com/science/article/pii/0360835296000472

import os
import numpy as np

import functions

# load data relatively to the current directory
current_path = os.getcwd()
print(current_path)
instance_path = os.path.join(current_path, "instances/instance_3.txt")
print(instance_path)

# load the problem instance
jobs = []
with open(instance_path, "r") as file:
    J, M = map(int, file.readline().split())
    for _ in range(J):
        line = file.readline()
        line = line.split()
        job = []
        for i in range(M):
            job.append((int(line[2*i]), int(line[2*i+1])))   # the form of (machine, processing_time)
            i += 1
        jobs.append(job)

print(f"J: {J}, M: {M}")
for job in jobs:
    print(job)

# prepare the GA
# 1) generate genom
# 2) generate population (list of genoms?) of some size
# 3) fitness function
# 4) selection function?
# 5) cross-over function
# 6) mutation

# chromosome = [1, 1, 2, 2, 0, 2, 0, 0, 1] # functions.generate_chromosome(J, M)
# print(f"chromosome: {chromosome}")

# fit = functions.fitness(J, M, chromosome, jobs)
# print(fit)


print(f"========= ALGORITHM ==========")
# algorithm
POPULATION_SIZE = 6
GENERATIONS_NUM = 20
if POPULATION_SIZE % 2 != 0:
    raise Exception("POPULATION_SIZE has to be even number!")
P = [functions.generate_chromosome(J, M) for _ in range(POPULATION_SIZE)]
# for p in P:
#     print(p)
P_fitness = [functions.fitness(J, M, p, jobs) for p in P]
# print(P_fitness)
current_best = P[np.array(P_fitness).argmin()]
# print(current_best)
for generation in range(GENERATIONS_NUM):
    P_prime = []
    for _ in range(POPULATION_SIZE / 2):   # random pairs? Every pair is removed from P from which those pairs are generated?
        # TODO selection
        c1, c2 = functions.selection()
        c1_prime, c2_prime = functions.crossover(J, M, c1 , c2)
        c1_mutated, c2_mutated = functions.mutation(J, M, c1_prime), functions.mutation(J, M, c2_prime)
        P_prime += [c1_mutated, c2_mutated]
    # TODO elitism - the P' is now of the same size as the P, so the P' is extended, or the worst chromosome in P' is replaced?
    P = P_prime
    P_fitness = [functions.fitness(J, M, p, jobs) for p in P]
    generation_best = P[np.array(P_fitness).argmin()]
    current_best = generation_best if generation_best <= current_best else current_best
    # we should implement 2 mutations, selections and crossovers for ONE chromosome representation?
    # Or do we have to come up with two different chromosome (problem) representations and implement
    # those three functions for those specific representations?