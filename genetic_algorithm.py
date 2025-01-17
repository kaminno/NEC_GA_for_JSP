# some inspiration taken from:
# https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=4f12bb2e8cdb0fd339b87f7c74234ea61b17d2d4
# https://www.kecl.ntt.co.jp/icl/as/members/yamada/ICGA91.PDF
# https://www.sciencedirect.com/science/article/pii/0360835296000472
# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7154916
# https://www.sciencedirect.com/science/article/pii/S0957417420302050

import os
import time
import random
import numpy as np

import functions

# load data relatively to the current directory
current_path = os.getcwd()
instance_path = os.path.join(current_path, "instances/instance_3.txt")

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
# for job in jobs:
#     print(job)

print(f"========= GENETIC ALGORITHM ==========")
ga_start = time.time()

POPULATION_SIZE = 100
GENERATIONS_NUM = 400
if POPULATION_SIZE % 2 != 0:
    raise Exception("POPULATION_SIZE has to be even number!")

P = [functions.generate_chromosome(J, M) for _ in range(POPULATION_SIZE)]
P_fitness = [functions.fitness(J, M, p, jobs) for p in P]
best_value = np.min(np.array(P_fitness))
best_chromosome = P[np.array(P_fitness).argmin()]

print(f"chromosome size: {len(best_chromosome)}")
print(f"In each generation, print the current best fitness")
print(f"initial state: {best_value}")

for generation in range(GENERATIONS_NUM):
    start = time.time()

    P_prime = []
    for _ in range(int(POPULATION_SIZE / 2)):
        # selection
        # c1 = functions.selection(J, M, P, jobs, "r")
        # c2 = functions.selection(J, M, P, jobs, "r")

        c1 = functions.selection(J, M, P, jobs, "t", 10)
        c2 = functions.selection(J, M, P, jobs, "t", 10)

        # cross-over
        # c1_prime, c2_prime = functions.crossover(J, M, c1 , c2)
        c1_prime, c2_prime = functions.crossover(J, M, c1 , c2, "t")

        # mutation
        # c1_mutated, c2_mutated = functions.mutation(J, M, c1_prime), functions.mutation(J, M, c2_prime)
        c1_mutated, c2_mutated = functions.mutation(J, M, c1_prime, "r", 0.1), functions.mutation(J, M, c2_prime, "r", 0.1)

        P_prime += [c1_mutated, c2_mutated]

    # TODO elitism?
    P = P_prime
    P_fitness = [functions.fitness(J, M, p, jobs) for p in P]

    # update global best
    generation_best = np.min(np.array(P_fitness))
    if generation_best < best_value:
        best_value = generation_best
        best_chromosome = P[np.array(P_fitness).argmin()]
        print(f"{generation} / {GENERATIONS_NUM}, best value: {best_value}")
    
    end = time.time()

ga_end = time.time()
print(f"Solution: {best_value} after {(ga_end - ga_start)/60:.2f} m")