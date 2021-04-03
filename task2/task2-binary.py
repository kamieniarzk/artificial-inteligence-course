import numpy as np
import random
import copy
import math
import sys

'''
26.03.2021

EARIN Exercise 2 - Evolutionary Algorithm

Jakub Szumski (295432)
Kacper Kamieniarz (293065)

'''


def evolution(population: np.array):

    best_overall_individual = population[0]
    best_overall_score = function(best_overall_individual)

    for iteration in range(iterations):
        selected_population = selection(population)
        crossed_population = crossover(selected_population)
        mutated_population = mutation(crossed_population)
        population = replacement(crossed_population, mutated_population)
        best_iteration_individual, best_iteration_score = fittest_individual(
            population)

        if verbose:
            print(
                f"================== Iteration: {iteration} ==================")
            print(
                f"Best score in this iteration was achived by: {best_iteration_individual} and is equal to: {best_iteration_score}")

        if best_iteration_score > best_overall_score:
            best_overall_individual = best_iteration_individual
            best_overall_score = best_iteration_score

    print("================== Final population ================")
    show_population(population)
    print(
        f"Best overall score was achived by: {best_overall_individual} and is equal to: {best_overall_score}")


def function(x: np.array) -> np.array:
    return c + b.T @ x + x.T @ A @ x


def selection(population: np.array):
    values = fitness_calculation(population)

    if values.min() != values.max():
        values = (values - values.min()) / (values.max() - values.min())
    else:
        values = np.full(values.shape[0], 1 / N)

    probabilities = values / values.sum()

    indices = np.random.choice(population.shape[0], size=N, p=probabilities)

    return np.array([population[idx] for idx in indices])


def crossover(population: np.array):
    retv = []

    assert len(
        population) % 2 == 0, "Population needs to be even in order to make pairs for crossover"

    for i in range(0, N - 1, 2):
        parent_1 = population[i]
        parent_2 = population[i+1]
        r = random.random()
        if r <= crossover_propability:
            for idx in range(len(parent_1)):
                crossover_point = random.randint(1, D - 1)
                parent_1[idx], parent_2[idx] = single_point_crossover(
                    parent_1[idx], parent_2[idx], crossover_point)

        retv += [parent_1, parent_2]

    return np.array(retv)


def mutation(population: np.array):
    retv = []
    for individual in population:
        retv.append(mutate(individual))

    return np.array(retv)


def replacement(x: np.array, y: np.array) -> np.array:
    k = len(y)
    return np.concatenate((y, x[k:]))


def fitness_calculation(population: np.array):
    fitness_scores = []

    for individual in population:
        fitness_scores.append(function(individual))

    return np.array(fitness_scores)


def generate_population():
    population = []

    for _ in range(N):
        individual = np.zeros((D,))
        for i in range(D):
            individual[i] = random.randint(i - 2**D, 2**D)
        population.append(individual)

    return np.array(population, dtype=np.int64)


def mutate(child: np.array):
    retv = np.copy(child)
    for idx, cell in enumerate(retv):
        binary_cell = decimal_to_binary(cell)
        copy_cell = ""
        for chromosome in binary_cell:
            r = random.random()
            if r <= mutation_probability:
                if chromosome == "0":
                    copy_cell += "1"
                else:
                    copy_cell += "0"
            else:
                copy_cell += chromosome
        retv[idx] = binary_to_decimal(copy_cell)
    return retv


def show_population(population: np.array):
    for idx, elem in enumerate(population):
        print(f"{idx}: {elem}")


def single_point_crossover(cell_1: int, cell_2: int, crossover_point: int):
    bits_cell_1 = decimal_to_binary(cell_1)
    bits_cell_2 = decimal_to_binary(cell_2)

    child_1 = bits_cell_1[:crossover_point] + bits_cell_2[crossover_point:]
    child_2 = bits_cell_2[:crossover_point] + bits_cell_1[crossover_point:]

    child_1 = binary_to_decimal(child_1)
    child_2 = binary_to_decimal(child_2)

    return child_1, child_2


def fittest_individual(population: np.array):
    max_value = np.max(fitness_calculation(population))
    best_idx = np.where(max_value == fitness_calculation(population))[0][0]
    print(f"Fittest indivitual: {population[best_idx]} with score {max_value}")
    return population[best_idx], max_value


def exit_with_error(msg: str):
    print(msg)
    sys.exit(1)


def twos_complement(val, bits):
    if (val & (1 << (bits - 1))) != 0:
        val = val - (1 << bits)
    return val


def decimal_to_binary(n: int):
    bits = D + 1
    return np.binary_repr(n, bits)


def binary_to_decimal(binary):
    unsigned = int(binary, 2)
    if binary[0] == "1":
        decimal = unsigned - 2**(D+1)
        return decimal
    else:
        return unsigned


if __name__ == '__main__':
    try:
        D = int(input("Specify d - number of dimensions: "))
        if D < 1:
            exit_with_error("D must be greater or equal to 1!")

        N = int(input("Specify N - population size: "))
        if N < 2:
            exit_with_error("Population must be ")

        c = float(input("Specify c constant: "))

        b = np.array([float(x) for x in input(
            "Place vector b separated by space: ").split(maxsplit=D - 1)])

        A = np.zeros(shape=(D, D))

        for row in range(D):
            A[row] = np.array([float(a) for a in input(
                f"Place row {row} of matrix A separated by space: ").split(maxsplit=D - 1)])

        crossover_propability = float(input("Input crossover probability: "))
        if crossover_propability < 0 or crossover_propability > 1:
            exit_with_error("Probability must be between 0 and 1!")

        mutation_probability = float(input("Input mutation probability: "))
        if mutation_probability < 0 or mutation_probability > 1:
            exit_with_error("Probability must be between 0 and 1!")

        iterations = int(input("Specify number of iterations: "))
        if iterations <= 0:
            exit_with_error("Iterations number must be bigger than 0!")

        print("Select verbosity mode")
        print("0 - No verbose")
        print("1 - Verbose")
        user_input = int(input("Mode: "))

        verbose = True if user_input == 1 else False

    except:
        exit_with_error("! Wrong Parameters !")

    population = generate_population()
    evolution(population)
