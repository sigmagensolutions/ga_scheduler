import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pprint
import csv

# Constants
DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
DAY_COUNT = 7
MAX_SHIFT_HOURS = 12
MIN_SHIFT_HOURS = 4
MIN_WORKERS_PER_WORKFORCE = 1
MAX_WORKERS_PER_WORKFORCE = 60
POPULATION_SIZE = 30
GENERATIONS = 100
MUTATION_PROB = 0.1
ELITE_COUNT = 3

# Demand curve settings
MINUTES_IN_WEEK = 7 * 24 * 60
AGG_INTERVAL = 10
AGG_COUNT = MINUTES_IN_WEEK // AGG_INTERVAL

# Generate a realistic weekly demand curve (smooth curve with peaks)
def generate_realistic_demand():
    curve = []
    for day in range(7):
        for minute in range(1440):
            hour = minute // 60
            if day in [0, 6]:
                base = 5 + 2 * np.sin((hour - 12) / 4)
            else:
                base = 10 + 5 * np.sin((hour - 12) / 3)
            noise = np.random.normal(0, 0.5)
            curve.append(max(0, base + noise))
    return curve

demand_curve = generate_realistic_demand()

def generate_random_worker():
    off_start = random.randint(0, DAY_COUNT - 1)
    days_off = [(off_start + i) % DAY_COUNT for i in range(2)]
    start_time = random.randint(0, 23)
    shift_length = random.randint(MIN_SHIFT_HOURS, MAX_SHIFT_HOURS)

    start_times = []
    shift_lengths = []
    for day in range(DAY_COUNT):
        if day in days_off:
            start_times.append('X')
            shift_lengths.append(0)
        else:
            start_times.append(start_time)
            shift_lengths.append(shift_length)

    return start_times + shift_lengths

def is_viable(worker):
    start_times = worker[:DAY_COUNT]
    shift_lengths = worker[DAY_COUNT:]
    worked_start_times = [start_times[i] for i in range(DAY_COUNT) if start_times[i] != 'X']
    worked_shift_lengths = [shift_lengths[i] for i in range(DAY_COUNT) if shift_lengths[i] > 0]
    if not worked_start_times or not worked_shift_lengths:
        return False
    if len(set(worked_start_times)) > 1 or len(set(worked_shift_lengths)) > 1:
        return False
    binary_days = [0 if s == 'X' else 1 for s in start_times] * 2
    for i in range(DAY_COUNT):
        if binary_days[i] == 0 and binary_days[i + 1] == 0:
            return True
    return False

def generate_random_workforce(max_workers=MAX_WORKERS_PER_WORKFORCE):
    workforce = []
    for _ in range(random.randint(MIN_WORKERS_PER_WORKFORCE, max_workers)):
        for _ in range(20):
            worker = generate_random_worker()
            if is_viable(worker):
                workforce.append(worker)
                break
    return workforce

def generate_population():
    return [generate_random_workforce() for _ in range(POPULATION_SIZE)]

def generate_capacity_curve(workforce):
    capacity = [0] * MINUTES_IN_WEEK
    for worker in workforce:
        start_times = worker[:DAY_COUNT]
        shift_lengths = worker[DAY_COUNT:]
        for day in range(DAY_COUNT):
            if start_times[day] == 'X':
                continue
            start_minute = day * 1440 + start_times[day] * 60
            end_minute = start_minute + shift_lengths[day] * 60
            for m in range(start_minute, min(end_minute, MINUTES_IN_WEEK)):
                capacity[m] += 1
    return capacity

def aggregate_curve(curve):
    return [sum(curve[i:i + AGG_INTERVAL]) / AGG_INTERVAL for i in range(0, len(curve), AGG_INTERVAL)]

def fitness(workforce, demand_curve):
    capacity = generate_capacity_curve(workforce)
    return sum((c - d) ** 2 for c, d in zip(capacity, demand_curve))

def mutate_worker(worker):
    if random.random() >= MUTATION_PROB:
        return worker
    w = worker.copy()
    idx = random.randint(0, DAY_COUNT - 1)
    if w[idx] != 'X':
        w[idx] = min(23, max(0, int(w[idx]) + random.choice([-1, 1])))
    shift_idx = DAY_COUNT + idx
    if w[shift_idx] != 0:
        w[shift_idx] = min(MAX_SHIFT_HOURS, max(MIN_SHIFT_HOURS, w[shift_idx] + random.choice([-1, 1])))
    return w if is_viable(w) else generate_random_worker()

def crossover(parent1, parent2):
    max_len = max(len(parent1), len(parent2))
    child = []
    for i in range(max_len):
        gene = None
        if i < len(parent1) and i < len(parent2):
            gene = random.choice([parent1[i], parent2[i]])
        elif i < len(parent1):
            gene = parent1[i] if random.random() < 0.5 else None
        elif i < len(parent2):
            gene = parent2[i] if random.random() < 0.5 else None
        if gene and is_viable(gene):
            child.append(gene)
    return child

def evolve_population_animated(population, demand_curve):
    fitness_log = []
    best_solution = None
    solution_per_generation = []
    first_gen_best_curve = []

    for gen in range(GENERATIONS):
        fitness_scores = [(fitness(wf, demand_curve), wf) for wf in population]
        fitness_scores.sort(key=lambda x: x[0])
        ranked = [wf for _, wf in fitness_scores]
        next_gen = ranked[:ELITE_COUNT]

        if gen == 0:
            first_gen_best_curve.extend(generate_capacity_curve(ranked[0]))

        while len(next_gen) < POPULATION_SIZE:
            parent1 = random.choices(ranked, weights=range(len(ranked), 0, -1))[0]
            parent2 = random.choices(ranked, weights=range(len(ranked), 0, -1))[0]
            max_rank = min(ranked.index(parent1), ranked.index(parent2)) + 1
            num_offspring = random.randint(1, max(1, POPULATION_SIZE // max_rank))

            for _ in range(num_offspring):
                if len(next_gen) >= POPULATION_SIZE:
                    break
                child = crossover(parent1, parent2)
                child = [mutate_worker(w) for w in child if is_viable(w)]
                next_gen.append(child)

        population = next_gen[:POPULATION_SIZE]
        best_fitness = fitness(population[0], demand_curve)
        best_curve = generate_capacity_curve(population[0])
        fitness_log.append(best_fitness)
        best_solution = population[0]
        solution_per_generation.append(best_solution)

        yield gen, best_fitness, best_curve, list(fitness_log), best_solution, first_gen_best_curve

    with open("best_solutions_by_generation.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Generation", "Worker_Index", "Start_Times", "Shift_Lengths"])
        for gen, sol in enumerate(solution_per_generation):
            for idx, worker in enumerate(sol):
                start_times = worker[:DAY_COUNT]
                shift_lengths = worker[DAY_COUNT:]
                writer.writerow([gen, idx, start_times, shift_lengths])

if __name__ == "__main__":
    population = generate_population()
    demand_agg = aggregate_curve(demand_curve)
    final_solution = []

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    x = list(range(len(demand_agg)))
    line1, = ax1.plot(x, demand_agg, label='Demand', color='black')
    line2, = ax1.plot(x, [0] * len(x), label='Capacity', color='blue')
    line_init, = ax1.plot(x, [0] * len(x), label='Initial Gen Best', color='gray', linestyle='--')
    ax1.set_title("Capacity vs Demand")
    ax1.legend()

    fit_x = list(range(GENERATIONS))
    fit_y = [None] * GENERATIONS
    line3, = ax2.plot(fit_x, fit_y, label='Fitness', color='red')
    ax2.set_title("Fitness Progression")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Fitness (Lower is Better)")
    ax2.legend()

    def update(data):
        gen, fit, capacity_curve, fitness_log, solution, first_gen_best_curve = data
        cap_agg = aggregate_curve(capacity_curve)
        line2.set_ydata(cap_agg)

        fit_x = list(range(len(fitness_log)))
        fit_y = fitness_log
        line3.set_xdata(fit_x)
        line3.set_ydata(fit_y)

        ax2.set_xlim(0, GENERATIONS)
        if fit_y:
            ymin, ymax = min(fit_y), max(fit_y)
            ax2.set_ylim(ymin * 0.95, ymax * 1.05)

        if gen == 0:
            init_agg = aggregate_curve(first_gen_best_curve)
            line_init.set_ydata(init_agg)

        global final_solution
        if gen == GENERATIONS - 1:
            final_solution = solution
        return line2, line3, line_init

    ani = animation.FuncAnimation(fig, update, evolve_population_animated(population, demand_curve), interval=200, repeat=False)
    plt.tight_layout()
    plt.show()

    print("\nBest Workforce (Final Generation):")
    pprint.pprint(final_solution)
