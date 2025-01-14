import random
import copy
import numpy as np

# use job-based representation; each number refers to the job and the occurence equals the order (2nd occurence means the second task etc)
def generate_chromosome(J, M):
    chromosome = [j for m in range(M) for j in range(J)]
    np.random.shuffle(chromosome)
    return chromosome

class Schedule:

    def __init__(self, J, M):
        self.J = J
        self.M = M
        self.times = [0 for j in range(self.J)] # each job's time when the last task is completed (and thus new task can begin)
        self.timetable = [[] for m in range(self.M)] # timetable for each machine
    
    def add(self, job, machine, process_time):
        machine_free_time = self.timetable[machine][-1][3] if len(self.timetable[machine]) > 0 else 0
        start_time = max(self.times[job], machine_free_time)
        self.times[job] = start_time + process_time
        self.timetable[machine].append((job, start_time, process_time, self.times[job]))

def fitness(J, M, chromosome, jobs):
    schedule = Schedule(J, M)
    order = [0 for _ in range(J)]
    for gen in chromosome:
        idx = order[gen]
        job = jobs[gen][idx]
        schedule.add(gen, job[0], job[1])
        order[gen] += 1
    return np.max(schedule.times)

def crossover(J, M, parent_1, parent_2, method="o"):
    if method == "o":
        return _one_point_crossover(J, M, parent_1, parent_2)
    elif method == "t":
        return _two_points_crossover(J, M, parent_1, parent_2)
    else:
        raise Exception("invalid selection method! @method parameter has to be from {'o', 't'}.")

def _one_point_crossover(J, M, parent_1, parent_2):
    # generate random crossover index
    cross_index = random.randrange(0, J*M)
    # print(f"\tcross-over index: {cross_index} / {J*M}, ")

    # split both parents on the index and copy the first part to their children
    child_1, child_2 = parent_1[0:cross_index], parent_2[0:cross_index]

    # store parent's remaining gens (and copy the list begining to make further operations easier)
    remaining_gens_1, remaining_gens_2 = parent_1[cross_index:] + parent_1[0:cross_index], parent_2[cross_index:] + parent_2[0:cross_index]

    # complete both children
    for gen in remaining_gens_2:
        child_1 += [gen] if child_1.count(gen) < M else []
    
    for gen in remaining_gens_1:
        child_2 += [gen] if child_2.count(gen) < M else []

    return child_1, child_2

def _two_points_crossover(J, M, parent_1, parent_2):
    print(f"Original chromosomes")
    print(parent_1)
    print(parent_2)
    # cross_index_1 = random.randrange(0, J*M)
    # cross_index_2 = random.randrange(0, J*M)
    cross_index_1 = 2
    cross_index_2 = 5
    print(f"Crossing interval")
    print(cross_index_1)
    print(cross_index_2)

    cross_part_1 = parent_1[cross_index_1: cross_index_2]
    cross_part_2 = parent_2[cross_index_1: cross_index_2]
    print("Crossing parts")
    print(cross_part_1)
    print(cross_part_2)

    parent_1[cross_index_1 : cross_index_2] = cross_part_2
    parent_2[cross_index_1 : cross_index_2] = cross_part_1
    print("Parts crossed")
    print(parent_1)
    print(parent_2)

    for i in range(len(cross_part_1)):
        # if jobs are different, find random corresponding number and swap it
        if cross_part_1[i] != cross_part_2[i]:
            occurance_to_swap_1 = random.randrange(0, M) + 1
            occurance_to_swap_2 = occurance_to_swap_1
            for j in range(len(parent_1)):
                if parent_1[j] == cross_part_2[i]:
                    occurance_to_swap_1 -= 1
                if parent_2[j] == cross_part_1[i]:
                    occurance_to_swap_2 -= 1
                if occurance_to_swap_1 == 0:
                    parent_1[j] = cross_part_2[i]
                if occurance_to_swap_2 == 0:
                    parent_2[j] = cross_part_1[i]
                if occurance_to_swap_1 == occurance_to_swap_2:
                    break
    
    print("Corrected children")
    print(parent_1)
    print(parent_2)
    return parent_1, parent_2

def mutation(J, M, chromosome, method="o", probability=0.5):
    if method == "o":
        return _one_permutation(J, M, chromosome)
    elif method == "r":
        return _random_permutation(J, M, chromosome, probability)
    else:
        raise Exception("invalid selection method! @method parameter has to be from {'o', 'r'}.")

def _one_permutation(J, M, chromosome):
    # generate random two indices
    m1, m2 = random.randrange(0, J*M), random.randrange(0, J*M)

    # swap gen on those indices
    tmp = chromosome[m1]
    chromosome[m1] = chromosome[m2]
    chromosome[m2] = tmp

    return chromosome

def _random_permutation(J, M, chromosome, probability):
    # random pairs of genes are selected, each gen exactly once and with some probability they are swaped

    indices = [i for i in range(J*M)]
    for pair in range(int(J*M / 2)):
        # randomly choose two gens which were not selected yet
        gen_1 = indices[random.randrange(0, len(indices))]
        indices.remove(gen_1)
        gen_2 = indices[random.randrange(0, len(indices))]
        indices.remove(gen_2)

        # swap genes if the probability is lower than the threshold
        prob = random.uniform(0, 1)
        if prob <= probability:
            tmp = chromosome[gen_1]
            chromosome[gen_1] = chromosome[gen_2]
            chromosome[gen_2] = tmp

    return chromosome


def selection(J, M, P, jobs, method="r", n=2):
    if method == "r":
        return _rank_selection(J, M, P, jobs)
    elif method == "t":
        return _tournament_selection(J, M, P, jobs, n)
    else:
        raise Exception("invalid selection method! @method parameter has to be from {'r', 't'}.")

def _rank_selection(J, M, P, jobs):
    # TODO check the correctness!

    # sort the population by fitness
    P_fitness = [fitness(J, M, p, jobs) for p in P]
    # print(P_fitness)

    sorted_fitness = copy.copy(P_fitness)
    sorted_fitness.sort(reverse=True)
    # print(sorted_fitness)

    ranks = [i + 1 for i in range(len(sorted_fitness))]
    # print(ranks)

    total_rank = np.array(ranks).sum()
    # print(total_rank)

    generated_rank = random.randrange(1, total_rank)
    # print(generated_rank)

    rank = 0
    for i in range(len(ranks)):
        rank += ranks[i]
        # print(f"i: {i}, rank: {rank}")
        if rank - generated_rank > 0:
            fitness_value = sorted_fitness[i]
            index = i
            # print(f"fitness: {fitness_value}")
            break
    
    # if there are more values of the same fitness
    fitness_count = sorted_fitness.count(fitness_value)
    if fitness_count > 1:
        first_index = sorted_fitness.index(fitness_value)
        predecesors_num = index - first_index
    
        fitness_processed = 0
        for i in range(len(P_fitness)):
            if P_fitness[i] == fitness_value:
                if predecesors_num == fitness_processed:
                    final_index = i
                else:
                    fitness_processed += 1
    else:
        final_index = index

    return P[final_index]
    # raise Exception("_rank_selection() method is not implemented yet, try another option.")

def _tournament_selection(J, M, P, jobs, n=2):
    P_fitness = [fitness(J, M, p, jobs) for p in P]
    participants = []
    while len(participants) != n:
        participant = random.randrange(0, len(P))
        if participants.count(participant) == 0:
            participants.append(participant)

    idx = None
    current_winner = np.inf
    for participant in participants:
        if P_fitness[participant] < current_winner:
            current_winner = P_fitness[participant]
            idx = participant
    
    winner = P[idx]

    return winner






def pick_neighbor(J, M, chromosome, n=1):
    # TODO define N(x, y)
    while n != 0:
        i1 = random.randrange(0, len(chromosome))
        i2 = random.randrange(0, len(chromosome))
        # if chromosome[i1] != chromosome[i2]:
        tmp = chromosome[i1]
        chromosome[i1] = chromosome[i2]
        chromosome[i2] = tmp
        n -= 1

    # return generate_chromosome(J, M)
    return chromosome

def E(fitness, reference):
    return fitness/reference

