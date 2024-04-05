import sys
import copy
import random
import numpy as np


def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def sort_points_by_distance(points):
    sorted_points = [points[0]]  # Start with the first point as the initial point

    while len(sorted_points) < len(points):
        current_point = sorted_points[-1]
        remaining_points = [point for point in points if point not in sorted_points]

        # Sort remaining points by distance to the current point
        remaining_points.sort(key=lambda x: euclidean_distance(current_point, x))

        # Add the closest point to the sorted list
        sorted_points.append(remaining_points[0])

    return sorted_points

class GA_MODEL:
    def __init__(self, **ga_params):
        self.depot_num = ga_params['depot_num']
        self.dp_num = ga_params['dp_num']
        self.customer_num = ga_params['customer_num']
        self.gene_list = None
        self.gene_num = None

        self.light_vehicle_load = None
        self.heavy_vehicle_load = None

        self.light_vehicle_cost = ga_params['light_vehicle_cost']
        self.heavy_vehicle_cost = ga_params['heavy_vehicle_cost']
        self.depot_cost = ga_params['depot_cost']
        self.dp_cost = ga_params['dp_cost']
        self.per_distance_cost = ga_params['per_distance_cost']

        self.population_num = ga_params['population_num']
        self.max_generations = ga_params['max_generations']
        self.tournament_size = ga_params['tournament_size']
        self.crossover_prob = ga_params['crossover_prob']
        self.mutation_prob = ga_params['mutation_prob']

    def load_gene_list(self, instance):
        self.gene_list = instance
        self.gene_num = len(instance)

        heavy_load_map = {
            37: 200,
            64: 150,
            120: 250,
            230: 300,
            340: 350,
        }
        self.light_vehicle_load = 40
        self.heavy_vehicle_load = heavy_load_map[len(instance)]

    def generate_route(self, customer_indices, dp_indices, depot_indices):
        gene_list = copy.deepcopy(self.gene_list)
        depot_list = [gene_list[i] for i in depot_indices]
        dp_list = [gene_list[i] for i in dp_indices]
        customer_list = [gene_list[i] for i in customer_indices]

        route_1 = []  # route_1 means the route from distribution points to customers
        choose_dp = []
        i = 0
        for dp in dp_list:
            dp_index = gene_list.index(dp)
            route_1.append(dp_index)

            remaining_capacity = dp[3]
            remaining_load = self.light_vehicle_load

            while remaining_capacity > 0 and i < self.customer_num:
                customer_demand = customer_list[i][2]
                remaining_capacity -= customer_demand
                remaining_load -= customer_demand
                if remaining_capacity > 0:
                    if remaining_load < 0:
                        route_1.append(dp_index)
                        remaining_load = self.light_vehicle_load - customer_demand
                    dp[2] += customer_demand  # accumulate the total demand of each route
                    customer_index = gene_list.index(customer_list[i])
                    route_1.append(customer_index)
                    i += 1
                    if i == self.customer_num:
                        route_1.append(dp_index)
                else:
                    route_1.append(dp_index)

            choose_dp.append(dp)
            if i >= self.customer_num:
                break

        route_2 = []  # route_2 means the route from depots to distribution points
        choose_depot = []
        i = 0
        for depot in depot_list:
            depot_index = gene_list.index(depot)
            route_2.append(depot_index)

            remaining_capacity = depot[3]
            remaining_load = self.heavy_vehicle_load

            while remaining_capacity > 0 and i < len(choose_dp):
                dp_demand = choose_dp[i][2]
                remaining_capacity -= dp_demand
                remaining_load -= dp_demand
                choose_dp[i][2] = 0
                if remaining_capacity > 0:
                    if remaining_load < 0:
                        route_2.append(depot_index)
                        remaining_load = self.heavy_vehicle_load - dp_demand
                    dp_index = gene_list.index(choose_dp[i])
                    route_2.append(dp_index)
                    i += 1
                    if i == len(choose_dp):
                        route_2.append(depot_index)
                else:
                    route_2.append(depot_index)

            choose_depot.append(depot)
            if i >= len(choose_dp):
                break

        return [route_1, route_2, choose_dp, choose_depot]

    def generate_chromosome(self):
        customer_index = random.sample(range(self.gene_num - self.customer_num, self.gene_num), self.customer_num)

        gene_list = copy.deepcopy(self.gene_list)
        depot_list = gene_list[0: self.depot_num]
        dp_list = gene_list[self.depot_num: self.depot_num + self.dp_num]
        customer_list = gene_list[self.depot_num + self.dp_num:]

        dp_cc = [sum(euclidean_distance(dp, i) for i in customer_list) for dp in dp_list]
        sorted_indices = sorted(enumerate(dp_cc, start=self.depot_num), key=lambda x: x[1])
        dp_index = [index for index, value in sorted_indices]

        sort_dp_list = [gene_list[i] for i in dp_index]

        depot_cc = [sum(euclidean_distance(depot, i) for i in sort_dp_list) for depot in depot_list]
        sorted_depot_indices = sorted(enumerate(depot_cc), key=lambda x: x[1])
        depot_index = [index for index, value in sorted_depot_indices]

        route_1, route_2, selected_dp, selected_depot = self.generate_route(customer_index, dp_index, depot_index)
        chromosome = [customer_index, dp_index, depot_index, route_1, route_2, selected_dp, selected_depot]

        return chromosome

    def generate_chromosome_0(self):
        gene_list = copy.deepcopy(self.gene_list)
        depot_list = gene_list[0: self.depot_num]
        dp_list = gene_list[self.depot_num: self.depot_num + self.dp_num]
        customer_list = gene_list[self.depot_num + self.dp_num:]

        dp_cc = [sum(euclidean_distance(dp, i) for i in customer_list) for dp in dp_list]
        sorted_indices = sorted(enumerate(dp_cc, start=self.depot_num), key=lambda x: x[1])
        dp_index = [index for index, value in sorted_indices]

        sort_dp_list = [gene_list[i] for i in dp_index]

        customer_index = []
        customer_index1 = []
        temp_list = []
        list2 = []
        for dp in sort_dp_list:
            customer_cc = [(euclidean_distance(dp, i), idx) for idx, i in
                           enumerate(customer_list, start=self.depot_num + self.dp_num)]
            customer_cc.sort(key=lambda x: x[0])

            for distance, idx in customer_cc:
                if idx not in customer_index1:
                    dp[3] -= gene_list[idx][2]
                    if dp[3] >= 0:
                        customer_index1.append(idx)
                        list2.append(idx)
                    else:
                        dp[3] += gene_list[idx][2]
                        temp_list.append(list2)
                        list2 = []
                        break
        temp_list.append(list2)

        for list1 in temp_list:
            node_list1 = [gene_list[i] for i in list1]
            sort_node_list1 = sort_points_by_distance(node_list1)
            for customer in sort_node_list1:
                customer_index.append(gene_list.index(customer))

        depot_cc = [sum(euclidean_distance(depot, i) for i in sort_dp_list) for depot in depot_list]
        sorted_depot_indices = sorted(enumerate(depot_cc), key=lambda x: x[1])
        depot_index = [index for index, value in sorted_depot_indices]

        route_1, route_2, selected_dp, selected_depot = self.generate_route(customer_index, dp_index, depot_index)
        chromosome = [customer_index, dp_index, depot_index, route_1, route_2, selected_dp, selected_depot]

        return chromosome
    def reconstruct_chromosome(self, customer_index, dp_index, depot_index):
        route_1, route_2, selected_dp, selected_depot = self.generate_route(customer_index, dp_index, depot_index)
        chromosome = [customer_index, dp_index, depot_index, route_1, route_2, selected_dp, selected_depot]

        return chromosome

    def evaluate_fitness(self, chromosome):
        route_1 = chromosome[3]
        route_2 = chromosome[4]
        selected_dp = chromosome[5]
        selected_depot = chromosome[6]
        depot_num = len(selected_depot)
        dp_num = len(selected_dp)

        facilities_cost = depot_num * self.depot_cost + dp_num * self.dp_cost

        selected_dp_index = [self.gene_list.index(dp) for dp in selected_dp]
        selected_depot_index = [self.gene_list.index(depot) for depot in selected_depot]
        light_vehicle_num = sum(route_1.count(dp_index) for dp_index in selected_dp_index) - dp_num
        heavy_vehicle_num = sum(route_2.count(depot_index) for depot_index in selected_depot_index) - depot_num
        vehicle_cost = light_vehicle_num * self.light_vehicle_cost + heavy_vehicle_num * self.heavy_vehicle_cost

        route1_distance = 0
        pre_node = self.gene_list[route_1[0]]
        for i in range(1, len(route_1)):
            cur_node = self.gene_list[route_1[i]]
            if pre_node[3] != 0 and cur_node[3] != 0:
                route1_distance += 0  # if previous gene and currant gene are dp, the distance is zero.
            else:
                route1_distance += euclidean_distance(pre_node, cur_node)
            pre_node = cur_node

        route2_distance = 0
        pre_node = self.gene_list[route_2[0]]
        for i in range(1, len(route_2)):
            cur_node = self.gene_list[route_2[i]]
            if self.gene_list.index(pre_node) < self.depot_num and self.gene_list.index(cur_node) < self.depot_num:
                route2_distance += 0  # if previous gene and currant gene is depot, the distance is zero.
            else:
                route2_distance += euclidean_distance(pre_node, cur_node)
            pre_node = cur_node

        distance_cost = (route1_distance + route2_distance) * self.per_distance_cost
        total_cost = facilities_cost + vehicle_cost + distance_cost

        return total_cost

    def tournament_selection(self, population):
        best_chromosome = None
        best_fitness = sys.float_info.max
        for i in range(self.tournament_size):
            index = random.randint(0, len(population) - 1)
            fitness = self.evaluate_fitness(population[index])
            if fitness < best_fitness:
                best_chromosome = population[index]
                best_fitness = fitness
        return best_chromosome

    def ordered_crossover(self, parent1, parent2):
        # customer gene has 0.75 probability crossover, otherwise 0.5 prob from parent1, 0.5 prob from parent2
        crossover_customer = [-1] * len(parent1[0])
        choose_method = random.random()
        if choose_method < 0.7:
            start_index, end_index = random.sample(range(0, len(parent1[0])), 2)
            if start_index > end_index:
                start_index, end_index = end_index, start_index

            for i in range(start_index, end_index):
                crossover_customer[i] = parent1[0][i]

            j = 0
            for i in range(len(parent2[0])):
                if j == start_index:
                    j = end_index
                if parent2[0][i] not in crossover_customer:
                    crossover_customer[j] = parent2[0][i]
                    j += 1
        else:
            crossover_customer = random.choice([parent1[0], parent2[0]])

        # dp gene has 0.5 probability crossover, otherwise 0.5 prob from parent1, 0.5 prob from parent2
        crossover_dp = [-1] * len(parent1[1])
        if 0.7 <= choose_method < 0.9:
            start_index, end_index = random.sample(range(0, len(parent1[1])), 2)
            if start_index > end_index:
                start_index, end_index = end_index, start_index

            for i in range(start_index, end_index):
                crossover_dp[i] = parent1[1][i]

            j = 0
            for i in range(len(parent2[1])):
                if j == start_index:
                    j = end_index
                if parent2[1][i] not in crossover_dp:
                    crossover_dp[j] = parent2[1][i]
                    j += 1
        else:
            crossover_dp = random.choice([parent1[1], parent2[1]])

        # depot gene has 0.25 probability crossover, otherwise 0.5 prob from parent1, 0.5 prob from parent2
        crossover_depot = [-1] * len(parent1[2])
        if choose_method >= 0.9:
            start_index, end_index = random.sample(range(0, len(parent1[2])), 2)
            if start_index > end_index:
                start_index, end_index = end_index, start_index

            for i in range(start_index, end_index):
                crossover_depot[i] = parent1[2][i]

            j = 0
            for i in range(len(parent2[2])):
                if j == start_index:
                    j = end_index
                if parent2[2][i] not in crossover_depot:
                    try:
                        crossover_depot[j] = parent2[2][i]
                        j += 1
                    except IndexError:
                        print('len(parent2[2]):', len(parent2[2]))
                        print('start_index:', start_index)
                        print('end_index:', end_index)
                        print('depot: ', crossover_depot)
                        print('parent2[2]: ', parent2[2])
                        print('j: ', j)
                        print('i: ', i)
        else:
            crossover_depot = random.choice([parent1[2], parent2[2]])

        crossover_chromosome = self.reconstruct_chromosome(crossover_customer, crossover_dp, crossover_depot)

        return crossover_chromosome

    def swap_mutation(self, chromosome):
        mutated_customer = chromosome[0]
        mutated_dp = chromosome[1]
        mutated_depot = chromosome[2]

        # customer gene mutation 0.9 probability
        if random.random() < 0.9:
            index1, index2 = random.sample(range(self.customer_num), 2)
            mutated_customer[index1], mutated_customer[index2] = mutated_customer[index2], mutated_customer[index1]

        # dp gene mutation 0.7 probability
        if random.random() < 0.7:
            index1, index2 = random.sample(range(self.dp_num), 2)
            mutated_dp[index1], mutated_dp[index2] = mutated_dp[index2], mutated_dp[index1]

        # depot gene mutation 0.5 probability
        if random.random() < 0.5:
            index1, index2 = random.sample(range(self.depot_num), 2)
            mutated_depot[index1], mutated_depot[index2] = mutated_depot[index2], mutated_depot[index1]

        mutated_chromosome = self.reconstruct_chromosome(mutated_customer, mutated_dp, mutated_depot)

        return mutated_chromosome

    def genetic_algorithm(self, instance):

        # initial gene_list
        self.load_gene_list(instance)

        # initial population
        population = [self.generate_chromosome() for _ in range(self.population_num-30)]
        for _ in range(30):
            population.append(self.generate_chromosome_0())

        # natural selection, survival of the fittest
        fitness_list = []
        for generation in range(self.max_generations):
            # selection
            parents = [self.tournament_selection(population) for _ in range(self.population_num)]

            # crossover
            offspring = []
            for i in range(0, self.population_num, 2):
                if i + 1 < self.population_num and random.random() < self.crossover_prob:
                    child1 = self.ordered_crossover(parents[i], parents[i + 1])
                    child2 = self.ordered_crossover(parents[i + 1], parents[i])
                    offspring.append(child1)
                    offspring.append(child2)
                else:
                    offspring.append(parents[i])
                    offspring.append(parents[i + 1])

            # mutation
            mutated_offspring = []
            for chromosome in offspring:
                if random.random() < self.mutation_prob:
                    mutated_chromosome = self.swap_mutation(chromosome)
                    mutated_offspring.append(mutated_chromosome)
                else:
                    mutated_offspring.append(chromosome)
            offspring = mutated_offspring

            # evaluate the population and update population
            for chromosome in offspring:
                fitness = self.evaluate_fitness(chromosome)
                if fitness < self.evaluate_fitness(population[-1]):
                    population[-1] = chromosome
                    population.sort(key=lambda x: self.evaluate_fitness(x))

            # record the best fitness of each generation
            best_fitness = self.evaluate_fitness(population[0])
            fitness_list.append(best_fitness)

        print(population[0])
        return population[0], self.evaluate_fitness(population[0])

