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
        start_time = max(self.times[job], machine_free_time) # task start time in the schedule
        self.times[job] = start_time + process_time
        self.timetable[machine].append((job, start_time, process_time, self.times[job]))

def fitness(J, M, chromosome, jobs):
    schedule = Schedule(J, M)
    # maintain number of processed tasks for each job
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
    children_1 = parent_1.copy()
    children_2 = parent_2.copy()

    # generate positions
    c1 = random.randrange(0, J*M)
    c2 = random.randrange(0, J*M)
    cross_index_1 = min(c1, c2)
    cross_index_2 = max(c1, c2)

    # get parts to swap
    cross_part_1 = parent_1[cross_index_1: cross_index_2]
    cross_part_2 = parent_2[cross_index_1: cross_index_2]

    # swap parts
    children_1[cross_index_1 : cross_index_2] = cross_part_2
    children_2[cross_index_1 : cross_index_2] = cross_part_1

    # check if both parts contains same number of same gens
    gen_count = parent_1.count(parent_1[0])
    for i in range(len(cross_part_1)):
        idx = cross_index_1 + i
        gen1 = children_1[idx]
        gen2 = children_2[idx]
        # if not, correct them. Each first occurence of overpresented gen is replaced by gen which is missing
        if gen1 != gen2:
            # find gen with wrong number of occurences
            if children_1.count(gen1) != gen_count:
                # solve bigger occurences
                if children_1.count(gen1) > gen_count:
                    for j in range(J):
                        if children_1.count(j) < gen_count:
                            children_1[children_1.index(gen1)] = j
                            break
                # solve smaller occurences
                elif children_1.count(gen1) < gen_count:
                    for j in range(J):
                        if children_1.count(j) > gen_count:
                            children_1[children_1.index(j)] = gen1
                            break

            # control the second chromosome
            if children_2.count(gen2) != gen_count:
                if children_2.count(gen2) > gen_count:
                    for j in range(J):
                        if children_2.count(j) < gen_count:
                            children_2[children_2.index(gen2)] = j
                            break
                elif children_2.count(gen2) < gen_count:
                    for j in range(J):
                        if children_2.count(j) > gen_count:
                            children_2[children_2.index(j)] = gen2
                            break

    return children_1, children_2

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
    for _ in range(int(J*M / 2)):
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
    P_fitness = [fitness(J, M, p, jobs) for p in P]

    # sort the population by fitness
    sorted_fitness = copy.copy(P_fitness)
    sorted_fitness.sort(reverse=True)

    # assign ranks
    ranks = [i + 1 for i in range(len(sorted_fitness))]

    total_rank = np.array(ranks).sum()

    generated_value = random.randrange(1, total_rank)

    rank = 0
    for i in range(len(ranks)):
        rank += ranks[i]
        
        # if the number is in the correct range, get the fitness
        if rank - generated_value > 0:
            fitness_value = sorted_fitness[i]
            index = i
            break
    
    # if there are more values of the same fitness, keeps the order. If the 2nd occurence was selected previously, 2nd chromosome of same fitness should be picked.
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

def _tournament_selection(J, M, P, jobs, n=2):
    # evaluate the population
    P_fitness = [fitness(J, M, p, jobs) for p in P]
    participants = []

    # choose n random indices (no duplicates are allowed)
    while len(participants) != n:
        participant = random.randrange(0, len(P))
        if participants.count(participant) == 0:
            participants.append(participant)

    # find chromosome with the best fitness
    idx = None
    current_winner = np.inf
    for participant in participants:
        if P_fitness[participant] < current_winner:
            current_winner = P_fitness[participant]
            idx = participant
    
    winner = P[idx]

    return winner

def pick_neighbor(J, M, chromosome, n=1):
    while n != 0:
        # generate random positions
        i1 = random.randrange(0, len(chromosome))
        i2 = random.randrange(0, len(chromosome))

        # swap values
        tmp = chromosome[i1]
        chromosome[i1] = chromosome[i2]
        chromosome[i2] = tmp

        n -= 1

    return chromosome
