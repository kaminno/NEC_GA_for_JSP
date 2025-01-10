# some inspiration taken from:
# https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=4f12bb2e8cdb0fd339b87f7c74234ea61b17d2d4
# https://www.kecl.ntt.co.jp/icl/as/members/yamada/ICGA91.PDF
# https://www.sciencedirect.com/science/article/pii/0360835296000472

import os

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

chromosome = [1, 1, 2, 2, 0, 2, 0, 0, 1] # functions.generate_chromosome(J, M)
print(f"chromosome: {chromosome}")

fit = functions.fitness(J, M, chromosome, jobs)
print(fit)


# algorithm
POPULATION_SIZE = 20
GENERATIONS_NUM = 20
P = [functions.generate_chromosome(J, M) for _ in range(20)]
P_fitness = [functions.fitness(J, M, p, jobs) for p in P]
# for p in P:
#     fit = functions.fitness(J, M, p, jobs)
#     print(f"{p} -> {fit}")
for generation in range(GENERATIONS_NUM):
    P_prime = []
