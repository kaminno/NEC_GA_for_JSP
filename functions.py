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
        # (job, start, duration, end)
        # print(self.timetable[machine])
        machine_free_time = self.timetable[machine][-1][3] if len(self.timetable[machine]) > 0 else 0
        # print(f"--------------------")
        # print(f"Machine {machine} ends on {machine_free_time}. Adding job {job} which can starts at {self.times[job]} with duration {process_time} will ends at {self.times[job] + process_time}")
        start_time = max(self.times[job], machine_free_time)
        self.times[job] = start_time + process_time
        # print(f"Job will start on machine at {start_time} and the updated machine will looks like {self.timetable[machine]}")
        self.timetable[machine].append((job, start_time, process_time, self.times[job]))

def fitness(J, M, chromosome, jobs):
    schedule = Schedule(J, M)
    # print(f"empty schedule: {schedule.timetable}")
    order = [0 for _ in range(J)]
    # print(f"order: {order}")
    for gen in chromosome:
        idx = order[gen]
        job = jobs[gen][idx]
        # print(f"job: {job}, idx: {idx}")
        schedule.add(gen, job[0], job[1])
        order[gen] += 1
    # print(f"order: {order}")

    # print(f"final schedule:")
    # for s in schedule.timetable:
    #     print(s)

    # print(f"ending times: {schedule.times}")
    
    return np.max(schedule.times)
