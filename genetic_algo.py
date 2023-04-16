import numpy as np
import random

# Define the fitness function
def fitness(solution):
    instances = 0
    num_classes = solution.shape[0]
    for c in range(1, num_classes):
        instances += np.sum(solution[c-1] == solution[c])
    return instances


def shuffle_labels(data, num_labels):
    data = np.array(data)
    unique_labels = np.arange(num_labels)

    shuffled_labels = np.random.permutation(unique_labels)
    label_mapping = dict(zip(unique_labels, shuffled_labels))
    data_new = np.vectorize(label_mapping.get)(data)

    return data_new

# Initialize a population
def init_population(initial_assignment, pop_size, num_classes, num_seats, num_labels):
    population = []
    for _ in range(pop_size):
        individual = initial_assignment.copy()
        for i in range(num_classes):
            individual[i, :, :] = shuffle_labels(individual[i, :, :], num_labels)
        population.append(individual)
    return np.array(population)

# Crossover
def crossover(parent1, parent2):
    num_classes, num_seats, _ = parent1.shape
    crossover_point = np.random.randint(num_classes)
    child = np.vstack((parent1[:crossover_point], parent2[crossover_point:]))
    return child

# Mutation
def mutate(child, mutation_rate, num_labels):
    for i in range(child.shape[0]):
        for label in range(num_labels):
            if random.random() < mutation_rate:
                new_label = random.randint(0, num_labels - 1)
                child[i][child[i] == label] = new_label
    return child

# Genetic Algorithm
def genetic_algorithm(initial_assignment, pop_size=10, num_classes=3, num_seats=7, num_labels=5, num_generations=100, mutation_rate=0.0):
    # Initialize population

    num_classes = initial_assignment.shape[0]
    num_seats = initial_assignment.shape[1]
    num_labels = np.amax(initial_assignment) + 1

    population = init_population(initial_assignment, pop_size, num_classes, num_seats, num_labels)

    for generation in range(num_generations):
        # Calculate fitness for each individual
        fitness_values = [fitness(solution) for solution in population]

        # Select the best individuals for reproduction
        sorted_indices = np.argsort(fitness_values)[::-1]
        parents = population[sorted_indices[:pop_size//2]]

        # Create the next generation using crossover and mutation
        next_gen = []
        for _ in range(pop_size):
            parent1, parent2 = random.sample(list(parents), 2)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate, num_labels)
            next_gen.append(child)

        population = np.array(next_gen)

    # Return the best solution found
    best_solution = population[np.argmax([fitness(solution) for solution in population])]
    return best_solution



initial_assignment = np.array([
    [
        [1, 1, 1, 1, 2, 2, 2, ],
        [1, 1, 1, 1, 2, 2, 2, ],
        [1, 1, 1, 1, 2, 3, 2, ],
        [1, 1, 1, 1, 3, 3, 3, ],
        [1, 1, 1, 1, 3, 3, 3, ],
        [2, 2, 2, 2, 3, 3, 3, ],
        [2, 2, 2, 2, 3, 4, 0, ]
    ],
    [
        [4, 4, 4, 4, 2, 2, 2, ],
        [4, 4, 4, 4, 2, 2, 2, ],
        [4, 4, 4, 4, 2, 3, 2, ],
        [4, 4, 4, 4, 1, 1, 1, ],
        [4, 4, 4, 4, 1, 1, 1, ],
        [2, 2, 2, 2, 1, 1, 1, ],
        [2, 2, 2, 2, 1, 1, 0, ]
    ],
    [
        [1, 1, 1, 1, 2, 2, 2, ],
        [1, 1, 1, 1, 2, 2, 2, ],
        [1, 1, 1, 1, 2, 3, 2, ],
        [1, 1, 1, 1, 3, 3, 3, ],
        [1, 1, 1, 1, 3, 3, 3, ],
        [2, 2, 2, 2, 3, 3, 3, ],
        [2, 2, 2, 2, 3, 3, 4, ]
    ],
    [
        [0, 0, 0, 1, 2, 2, 2, ],
          [0, 0, 0, 1, 2, 2, 2, ],
          [0, 0, 0, 1, 2, 3, 2, ],
          [0, 0, 0, 1, 1, 1, 1, ],
          [1, 0, 0, 1, 1, 1, 1, ],
          [2, 2, 2, 2, 1, 1, 3, ],
          [2, 2, 2, 2, 3, 3, 4 ]
    ]
])
initial_assignment = np.random.randint(30, size=(15, 200, 200))

# initial_assignment = np.array([
#     [
#         [0, 1, 2, 3, 4],
#         [1, 0, 3, 2, 4],
#         [2, 3, 0, 1, 4],
#         [3, 2, 1, 0, 4],
#         [4, 4, 4, 4, 4]
#     ],
#     [
#         [0, 0, 0, 0, 0],
#         [1, 1, 1, 1, 1],
#         [2, 2, 2, 2, 2],
#         [3, 3, 3, 3, 3],
#         [4, 4, 4, 4, 4]
#     ],
#     [
#         [0, 0, 0, 0, 0],
#         [1, 1, 1, 1, 1],
#         [2, 2, 1, 2, 2],
#         [3, 3, 1, 3, 3],
#         [4, 4, 1, 4, 4]
#     ],
# ])

# Run the genetic algorithm
final_assignment = genetic_algorithm(initial_assignment,  pop_size=1000, num_generations=2)
print('-------------')

print('final')
print(final_assignment)
