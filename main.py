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
instance_path = os.path.join(current_path, "instances/instance_1.txt")

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

# ch1 = [1, 2, 3, 1, 2, 3]
# ch2 = [1, 3, 2, 1, 2, 3]
# functions.crossover(2, 3, ch1, ch2, "t")

# raise Exception(f"In dev")

print(f"========= GENETIC ALGORITHM ==========")
POPULATION_SIZE = 50
GENERATIONS_NUM = 500
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
    # print(f"----------------\ngeneration: {generation}, ", end="")
    # print(f"generation: {generation}, ", end="")
    for _ in range(int(POPULATION_SIZE / 2)):
        # selection
        # c1 = functions.selection(J, M, P, jobs, "r")
        # c2 = functions.selection(J, M, P, jobs, "r")

        c1 = functions.selection(J, M, P, jobs, "t", 10)
        c2 = functions.selection(J, M, P, jobs, "t", 10)

        # cross-over
        c1_prime, c2_prime = functions.crossover(J, M, c1 , c2)

        # mutation
        c1_mutated, c2_mutated = functions.mutation(J, M, c1_prime), functions.mutation(J, M, c2_prime)
        # c1_mutated, c2_mutated = functions.mutation(J, M, c1_prime, "r", 0.1), functions.mutation(J, M, c2_prime, "r", 0.1)
        P_prime += [c1_mutated, c2_mutated]
    # TODO elitism - the P' is now of the same size as the P, so the P' is extended, or the worst chromosome in P' is replaced?
    P = P_prime
    P_fitness = [functions.fitness(J, M, p, jobs) for p in P]

    # update global best
    generation_best = np.min(np.array(P_fitness))
    if generation_best < best_value:
        best_value = generation_best
        best_chromosome = P[np.array(P_fitness).argmin()]
        print(f"{generation} / {GENERATIONS_NUM}, best value: {best_value}")
    
    end = time.time()

    # print(f"best value: {best_value}, time: {(end - start):.2f} s")

# print(f"========= SIMULATED ANNEALING ==========")
# T_0 = 100
# T_I = 1
# T = T_0 - 1
# I = 100000
# s = functions.generate_chromosome(J, M)
# s_fitness = functions.fitness(J, M, s, jobs)
# best_solution = s_fitness
# # while T != 0:
# for i in range(I):
#     # print(f"T: {T:.2f}, ", end="")
#     # s_prime = functions.pick_neighbor(J, M, s, n=int(len(s)/10))
#     s_prime = functions.pick_neighbor(J, M, s, n=5)
#     s_prime_fitness = functions.fitness(J, M, s_prime, jobs)
#     # print(f"E_current: {s_fitness}, E_new: {s_prime_fitness}, ", end="")
#     if s_prime_fitness <= s_fitness:
#         s = s_prime
#         s_fitness = s_prime_fitness
#     else:
#         # print(f"T = {T}, {functions.E(s_fitness, s_fitness):.2f}, {functions.E(s_prime_fitness, s_fitness):.2f}, {(functions.E(s_prime_fitness, s_fitness) - functions.E(s_fitness, s_fitness)):.2f}, {(-(functions.E(s_prime_fitness, s_fitness) - functions.E(s_fitness, s_fitness))/T):.5f}", end="")
#         # print(f"current: {s_fitness}, new: {s_prime_fitness} -> {s_fitness - s_prime_fitness}")
#         p_top = np.exp((s_fitness - s_prime_fitness) / T)
#         p = random.uniform(0, 1)
#         if p <= p_top:
#             s = s_prime
#             s_fitness = s_prime_fitness
#         # print(f"{p:.4f} <= {p_top:.4f}, ", end="")
#     if s_fitness < best_solution:
#         best_solution = s_fitness
#         print(f"i: {i}, new best: {s_fitness}")
#     # print(f"new best: {s_fitness}")
#     T = T - (T_0 - T_I)/I

# print(f"Final solution: {best_solution}")