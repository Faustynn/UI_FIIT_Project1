import random
import ast
import configparser
import copy
import time
import matplotlib.pyplot as plt


def plot_fitness_evolution(max_fitnesses, avg_fitnesses):
    generations = range(1, len(max_fitnesses) + 1)
    plt.plot(generations, max_fitnesses, label='Max Fitness')
    plt.plot(generations, avg_fitnesses, label='Avg Fitness')

    for i, (max_fitness, avg_fitness) in enumerate(zip(max_fitnesses, avg_fitnesses)):
        plt.annotate(f'{max_fitness}', (generations[i], max_fitness), textcoords="offset points", xytext=(0,10), ha='center')
        plt.annotate(f'{avg_fitness}', (generations[i], avg_fitness), textcoords="offset points", xytext=(0,-15), ha='center')

    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title('Statistic Fitness Evolution in Generations')
    plt.legend()
    plt.grid(True)
    plt.show()

def print_population(population, garden,debug):
    for idx, individ in enumerate(population):
        start_position, gene_path = individ
        print(f"Individual {idx + 1}:")
        print(f"  *-Start Position: {start_position}")
        print(f"  *-Gene Path: {gene_path}")

        score = fitness_metod(individ, garden, debug)
        print(f"  *-Score: {score}")
        print("###################################################################################################")
def print_garden(garden, current_position=None):
    cols = len(garden[0])

    print()
    print("    " + " ".join(f"{i:2}" for i in range(cols)))
    print("  +" + "---" * cols + "+")

    for y, row in enumerate(garden):
        row_display = []
        for x, value in enumerate(row):
            if current_position == (y, x):
                row_display.append(" *")
            else:
                row_display.append(f"{value:2}")
        print(f"{y:2}| " + " ".join(row_display) + " |")
    print("  +" + "---" * cols + "+")

def init_matrix(columns, rows, stones_pos):
    garden_matrix = [[0 for _ in range(columns)] for _ in range(rows)]
    for stone in stones_pos:
        row, col = stone
        garden_matrix[row][col] = 2
    return garden_matrix

def generate_genes_position(garden, n):
    rows = len(garden)
    cols = len(garden[0])

    start_garden_pos = [((0, col),'S') for col in range(cols)]  #up
    start_garden_pos += [((rows - 1, col),'N') for col in range(cols)]  #down
    start_garden_pos += [((row, 0),'E') for row in range(1, rows - 1)]  #left
    start_garden_pos += [((row, cols - 1),'W') for row in range(1, rows - 1)]  #right

    random_positions = random.sample(start_garden_pos, n)
    return tuple(random_positions for _ in range(n))
def generate_gene_path(n): # 0 - right, 1 - left
    return tuple(random.choice([0, 1]) for _ in range(n))

def init_individual(genes_for_position, genes_for_path, garden):
    start_position = random.choice(generate_genes_position(garden, genes_for_position))
    gene_path = generate_gene_path(genes_for_path)
    return start_position, gene_path
def init_population(population_size, genes_for_position, genes_for_path, garden):
    individuals = [init_individual(genes_for_position, genes_for_path, garden) for _ in range(population_size)]
    return individuals


direction = ['N', 'E', 'S', 'W']
def move(x, y, direct):
    if direct == 'N':
        return y - 1, x  # up
    elif direct == 'E':
        return y, x + 1  # right
    elif direct == 'S':
        return y + 1, x  # down
    elif direct == 'W':
        return y, x - 1  # left
def turn_right(current_direction):
    idx = direction.index(current_direction)
    return direction[(idx + 1) % 4]
def turn_left(current_direction):
    idx = direction.index(current_direction)
    return direction[(idx - 1) % 4]

def fitness_metod(person, matrix, debug):
    garden = copy.deepcopy(matrix)
    start_positions, gene_path = person
    index_pos = 0
    index_gene = 0
    (y, x), D = start_positions[index_pos]
    path_history = []
    current_direction = D
    score = 0
    all_directions_blocked = True
    direction_attempts = {directions: 0 for directions in range(len(gene_path))}

    def debug_print(message):
        if debug:
            print(message)

    if garden[y][x] in (1, 2):
        debug_print("I cant start in this position: " + str((y, x)))
        index_pos += 1
        (y, x), D = start_positions[index_pos]

    while index_pos < len(start_positions):
        if index_gene >= len(gene_path):
            index_gene = 0

        if garden[y][x] == 0:
            debug_print("I can move in " + str((y, x)) + ", continue.")
            garden[y][x] = 1
            score += 1
            path_history.append(((y, x), current_direction))
            (y, x) = move(x, y, current_direction)
            if debug:
                print_garden(garden)
            direction_attempts[index_gene] = 0

        elif garden[y][x] in (1, 2):
            ((prev_y, prev_x), prev_direction) = path_history[-1]
            MAX_CYCLES = 4
            cycle_count = 0

            while cycle_count < MAX_CYCLES:
                if index_gene >= len(gene_path):
                    index_gene = 0

                if gene_path[index_gene] == 0:  # направо
                    current_direction = turn_right(current_direction)
                    debug_print("Turn right: " + str(current_direction))
                elif gene_path[index_gene] == 1:  # налево
                    current_direction = turn_left(current_direction)
                    debug_print("Turn left: " + str(current_direction))
                else:
                    print("Error: invalid gene")
                    break

                (temp_y, temp_x) = move(prev_x, prev_y, current_direction)

                if temp_y < 0 or temp_y >= len(garden) or temp_x < 0 or temp_x >= len(garden[0]):
                    index_pos += 1
                    break

                if garden[temp_y][temp_x] == 0:  # Успешный шаг
                    y, x = temp_y, temp_x
                    path_history.append(((y, x), current_direction))
                    index_gene += 1
                    debug_print("Go to " + str((y, x)) + " position.")
                    all_directions_blocked = False
                    direction_attempts[index_gene - 1] = 0 if index_gene > 0 else 0
                    if debug:
                        print_garden(garden)
                    break
                else:
                    debug_print(f"Hit in ({temp_y}, {temp_x}), return to ({prev_y}, {prev_x}).")
                    y, x = prev_y, prev_x
                    current_direction = prev_direction
                    cycle_count += 1
                    index_gene += 1

                    if debug:
                        print_garden(garden)



            if all_directions_blocked:
                debug_print(f"All direction around ({prev_y}, {prev_x}) blocked. GAME OVER")
                if debug:
                    print_garden(garden)
                break

            if index_gene > 0:
                direction_attempts[index_gene - 1] += 1

            if index_gene > 0 and direction_attempts[index_gene - 1] > len(gene_path):
                debug_print("Alot of try to go to similar position. GAME OVER")
                if debug:
                    print_garden(garden)
                break

        else:
            print("Error")
            break

        if y < 0 or y >= len(garden) or x < 0 or x >= len(garden[0]):
            debug_print("Out of garden. Go to another start position.")

            while index_pos < len(start_positions):
                (y, x), N = start_positions[index_pos]

                if garden[y][x] == 1 or garden[y][x] == 2:
                    debug_print(f"Position ({y}, {x}) is blocked. Try another.")
                    index_pos += 1
                else:
                    current_direction = N
                    debug_print("Start with new position: " + str((y, x)) + " with direction " + str(current_direction))
                    break

            if index_pos >= len(start_positions):
                debug_print("All start positions are blocked. GAME OVER")
                break

    return score, garden

def tournament_selection(population, garden ,tournament_size,debug):
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(population, tournament_size)
        tournament_sorted = sorted(tournament, key=lambda individual: fitness_metod(individual, garden,debug), reverse=True)
        best_individ = tournament_sorted[0]
        selected.append(best_individ)
    return selected

def crossover(mother, father, rate):
    if random.random() > rate:
        return mother


    mother_start_pos, mother_gene_path = mother
    father_start_pos, father_gene_path = father


    crossover_point_pos = random.randint(1, len(mother_start_pos) - 1)
    crossover_point_path = random.randint(1, len(mother_gene_path) - 1)

    child_start_pos = mother_start_pos[:crossover_point_pos] + father_start_pos[crossover_point_pos:]
    child_gene_path = mother_gene_path[:crossover_point_path] + father_gene_path[crossover_point_path:]

    return child_start_pos, child_gene_path

# Mutation
def mutation_gene_path(person, rate):
    start_positions, gene_path = person

    mutated_gene_path = list(gene_path)
    for i in range(len(mutated_gene_path)):
        if random.random() < rate:
            mutated_gene_path[i] = random.choice([0, 1])

    return start_positions, tuple(mutated_gene_path)
def mutation_position_path(person, rate,matrix):
    start_positions, gene_path = person

    mutated_start_positions = list(gene_path)
    for i in range(len(mutated_start_positions)):
        if random.random() < rate:
            mutated_start_positions[i] = random.choice(generate_genes_position(matrix, 1))

    return tuple(mutated_start_positions), gene_path



def main():
    config = configparser.ConfigParser()
    config_path = 'config/config.properties'
    config.read(config_path)


    height = config.getint('start-up-config', 'rows')
    width = config.getint('start-up-config', 'colum')
    stones = config.getint('start-up-config', 'stones')
    stones_position = ast.literal_eval(config.get('start-up-config', 'stones_pos'))

    population_size = config.getint('genetic-settings', 'pop_size')
    mutation_rate1 = config.getfloat('genetic-settings', 'mutation_rate1')
    mutation_rate2 = config.getfloat('genetic-settings', 'mutation_rate2')
    crossover_rate = config.getfloat('genetic-settings', 'cross_rate')
    max_generations = config.getint('genetic-settings', 'max_gen')
    tournament_size = config.getint('genetic-settings', 'tournament_size')
    debug_info = config.getboolean('genetic-settings', 'debug_info')

    max_score = (height * width) - stones
    genes_for_position = int((height + width))
    genes_for_path = int(stones)

    random_seed = config.getint('genetic-settings', 'random_seed')
    if debug_info:
        random.seed(random_seed)

    # init garden
    garden = init_matrix(width, height, stones_position)

    # init population
    population = init_population(population_size, genes_for_position, genes_for_path, garden)

    # init statistic parameters
    max_fitnesses = []
    avg_fitnesses = []

    # začneme počítat čas
    start_time = time.time()

    # Trening loop
    for generation in range(max_generations):
        print(f"------------Generation {generation + 1} ------------")

        population_scores = [(individual, fitness_metod(individual, garden,debug_info)) for individual in population]

        scores = [score[1][0] for score in population_scores]
        max_fitness = max(scores)
        avg_fitness = sum(scores) / len(scores)
        max_fitnesses.append(max_fitness)
        avg_fitnesses.append(avg_fitness)

        # tournament for population
        selected_population = tournament_selection(population, garden,tournament_size,debug_info)

        # new population
        new_population = []
        for i in range(0, len(selected_population), 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i + 1] if i + 1 < len(selected_population) else selected_population[0]
            child1 = crossover(parent1, parent2, crossover_rate)
            child2 = crossover(parent2, parent1, crossover_rate)

            child1 = mutation_gene_path(child1, mutation_rate1)
            child2 = mutation_gene_path(child2, mutation_rate2)

            new_population.extend([child1, child2])

        # replace olp pop. to new one
        population = new_population[:population_size]
        print_population(population, garden, debug_info)

    end_time = time.time()
    elapsed_time = end_time - start_time
    best_individual = max(population_scores, key=lambda x: x[1])

    # print results
    fitness_metod(best_individual[0], garden,True)
    print("##################################################################################################")
    print(f"Running time: {elapsed_time:.2f} sec")
    print(f"Best individ in last generation: {best_individual[0][0]}")
    print(f"With score {best_individual[1][0]} из {max_score}")
    print()
    plot_fitness_evolution(max_fitnesses, avg_fitnesses)

if __name__ == "__main__":
    main()