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

# print(f"J: {J}, M: {M}")
# for job in jobs:
#     print(job)

print(f"========= SIMULATED ANNEALING ==========")
alg_start = time.time()

T_0 = 100
T_I = 1
T = T_0 - 1
I = 1000000

# init values
s = functions.generate_chromosome(J, M)
s_fitness = functions.fitness(J, M, s, jobs)

# remember best solution
best_solution = s_fitness

for i in range(I):
    # pick and evaluate new state
    s_prime = functions.pick_neighbor(J, M, s, n=10)
    s_prime_fitness = functions.fitness(J, M, s_prime, jobs)

    # if it is better, keep it
    if s_prime_fitness <= s_fitness:
        s = s_prime
        s_fitness = s_prime_fitness
    else:
        # if not, keep it with some probability
        p_top = np.exp((s_fitness - s_prime_fitness) / T)
        p = random.uniform(0, 1)
        if p <= p_top:
            s = s_prime
            s_fitness = s_prime_fitness
    
    # if the new state is better, update the best solution
    if s_fitness < best_solution:
        best_solution = s_fitness
        print(f"i: {i}, new best: {s_fitness}")

    # update T
    T = T - (T_0 - T_I)/I

alg_end = time.time()
print(f"Final solution: {best_solution} after {(alg_end - alg_start)/60:.2f} m")