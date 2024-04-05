import random
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import scipy.special


def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

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


class SA_MODEL:
    def __init__(self, **sa_params):
        self.depot_num = sa_params['depot_num']
        self.dp_num = sa_params['dp_num']
        self.customer_num = sa_params['customer_num']
        self.node_list = None
        self.node_num = None

        self.light_vehicle_load = None
        self.heavy_vehicle_load = None

        self.light_vehicle_cost = sa_params['light_vehicle_cost']
        self.heavy_vehicle_cost = sa_params['heavy_vehicle_cost']
        self.depot_cost = sa_params['depot_cost']
        self.dp_cost = sa_params['dp_cost']
        self.per_distance_cost = sa_params['per_distance_cost']

        self.init_temperature = sa_params['init_temperature']
        self.cooling_rate = sa_params['cooling_rate']
        self.max_iteration = sa_params['max_iteration']

    def load_node_list(self, instance):
        self.node_list = instance
        self.node_num = len(instance)

        heavy_load_map = {
            37: 200,
            64: 150,
            120: 250,
            340: 350,
        }
        self.light_vehicle_load = 40
        self.heavy_vehicle_load = heavy_load_map[len(instance)]

    def generate_route(self, customer_indices, dp_indices, depot_indices):
        node_list = copy.deepcopy(self.node_list)
        depot_list = [node_list[i] for i in depot_indices]
        dp_list = [node_list[i] for i in dp_indices]
        customer_list = [node_list[i] for i in customer_indices]

        route_1 = []  # route_1 means the route from distribution points to customers
        choose_dp = []
        i = 0
        for dp in dp_list:
            dp_index = node_list.index(dp)
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
                    customer_index = node_list.index(customer_list[i])
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
            depot_index = node_list.index(depot)
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
                    dp_index = node_list.index(choose_dp[i])
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

    def initial_solution(self):
        node_list = copy.deepcopy(self.node_list)
        depot_list = node_list[0: self.depot_num]
        dp_list = node_list[self.depot_num: self.depot_num + self.dp_num]
        customer_list = node_list[self.depot_num + self.dp_num: ]

        dp_cc = [sum(euclidean_distance(dp, i) for i in customer_list) for dp in dp_list]
        sorted_indices = sorted(enumerate(dp_cc, start=self.depot_num), key=lambda x: x[1])
        dp_index = [index for index, value in sorted_indices]

        sort_dp_list = [node_list[i] for i in dp_index]

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
                    dp[3] -= node_list[idx][2]
                    if dp[3] >= 0:
                        customer_index1.append(idx)
                        list2.append(idx)
                    else:
                        dp[3] += node_list[idx][2]
                        temp_list.append(list2)
                        list2 = []
                        break
        temp_list.append(list2)

        for list1 in temp_list:
            node_list1 = [node_list[i] for i in list1]
            sort_node_list1 = sort_points_by_distance(node_list1)
            for customer in sort_node_list1:
                customer_index.append(node_list.index(customer))

        depot_cc = [sum(euclidean_distance(depot, i) for i in sort_dp_list) for depot in depot_list]
        sorted_depot_indices = sorted(enumerate(depot_cc), key=lambda x: x[1])
        depot_index = [index for index, value in sorted_depot_indices]

        route_1, route_2, selected_dp, selected_depot = self.generate_route(customer_index, dp_index, depot_index)
        initial_solution = [customer_index, dp_index, depot_index, route_1, route_2, selected_dp, selected_depot]

        return initial_solution

    def reconstruct_solution(self, customer_index, dp_index, depot_index):
        route_1, route_2, selected_dp, selected_depot = self.generate_route(customer_index, dp_index, depot_index)
        solution = [customer_index, dp_index, depot_index, route_1, route_2, selected_dp, selected_depot]

        return solution

    def generate_neighbor_solution(self, solution):
        solutions = copy.deepcopy(solution)
        index_customer = solutions[0]
        index_dp = solutions[1]
        index_depot = solutions[2]

        choose_method = random.random()
        if choose_method < 1/3:
            # insert
            prob = random.random()
            if prob < 0.7:
                index1, index2 = random.sample(range(self.customer_num), 2)
                index_customer.insert(index2, index_customer.pop(index1))
            elif 0.7 <= prob < 0.9:
                index1, index2 = random.sample(range(self.dp_num), 2)
                index_dp.insert(index2, index_dp.pop(index1))
            else:
                index1, index2 = random.sample(range(self.depot_num), 2)
                index_depot.insert(index2, index_depot.pop(index1))

        elif choose_method >= 2/3:
            # swap
            prob = random.random()
            if prob < 0.75:
                index1, index2 = random.sample(range(self.customer_num), 2)
                index_customer[index1], index_customer[index2] = index_customer[index2], index_customer[index1]
            elif 0.75 <= prob < 0.9:
                index1, index2 = random.sample(range(self.dp_num), 2)
                index_dp[index1], index_dp[index2] = index_dp[index2], index_dp[index1]
            else:
                index1, index2 = random.sample(range(self.depot_num), 2)
                index_depot[index1], index_depot[index2] = index_depot[index2], index_depot[index1]
        else:
            # 2-opt move
            if random.random() < 0.8:
                index1, index2 = random.sample(range(self.customer_num), 2)
                if index1 > index2:
                    index1, index2 = index2, index1
                index_customer = index_customer[:index1] + index_customer[index1:index2][::-1] + index_customer[index2:]
            else:
                index1, index2 = random.sample(range(self.dp_num), 2)
                if index1 > index2:
                    index1, index2 = index2, index1
                index_dp = index_dp[:index1] + index_dp[index1:index2][::-1] + index_dp[index2:]

        neighbor_solution = self.reconstruct_solution(index_customer, index_dp, index_depot)

        return neighbor_solution

    def evaluate_solution(self, solution):
        route_1 = solution[3]
        route_2 = solution[4]
        selected_dp = solution[5]
        selected_depot = solution[6]
        depot_num = len(selected_depot)
        dp_num = len(selected_dp)

        facilities_cost = depot_num * self.depot_cost + dp_num * self.dp_cost

        selected_dp_index = [self.node_list.index(dp) for dp in selected_dp]
        selected_depot_index = [self.node_list.index(depot) for depot in selected_depot]
        light_vehicle_num = sum(route_1.count(dp_index) for dp_index in selected_dp_index) - dp_num
        heavy_vehicle_num = sum(route_2.count(depot_index) for depot_index in selected_depot_index) - depot_num
        vehicle_cost = light_vehicle_num * self.light_vehicle_cost + heavy_vehicle_num * self.heavy_vehicle_cost

        route1_distance = 0
        pre_node = self.node_list[route_1[0]]
        for i in range(1, len(route_1)):
            cur_node = self.node_list[route_1[i]]
            if pre_node[3] != 0 and cur_node[3] != 0:
                route1_distance += 0  # if previous gene and currant gene are dp, the distance is zero.
            else:
                route1_distance += euclidean_distance(pre_node, cur_node)
            pre_node = cur_node

        route2_distance = 0
        pre_node = self.node_list[route_2[0]]
        for i in range(1, len(route_2)):
            cur_node = self.node_list[route_2[i]]
            if self.node_list.index(pre_node) < self.depot_num and self.node_list.index(cur_node) < self.depot_num:
                route2_distance += 0  # if previous gene and currant gene is depot, the distance is zero.
            else:
                route2_distance += euclidean_distance(pre_node, cur_node)
            pre_node = cur_node

        distance_cost = (route1_distance + route2_distance) * self.per_distance_cost
        total_cost = facilities_cost + vehicle_cost + distance_cost

        return total_cost

    def simulated_annealing(self, instance):

        # initial node list
        self.load_node_list(instance)

        # initialize
        temperature = self.init_temperature
        current_solution = self.initial_solution()
        best_solution = copy.deepcopy(current_solution)

        # annealing procedure
        solution_score_list = []
        for iteration in range(self.max_iteration):
            # randomly generating neighboring solutions
            neighbor_solution = self.generate_neighbor_solution(current_solution)

            # compute cost difference
            current_cost = self.evaluate_solution(current_solution)
            neighbor_cost = self.evaluate_solution(neighbor_solution)
            cost_difference = neighbor_cost - current_cost

            # accept new solution probability
            # acceptance_probability = scipy.special.expit(- cost_difference / temperature)
            acceptance_probability = np.exp(- cost_difference / temperature)

            # accept new solution based on probability
            if cost_difference < 0 or random.random() < acceptance_probability:
                current_solution = copy.deepcopy(neighbor_solution)

            # update best solution
            if self.evaluate_solution(current_solution) < self.evaluate_solution(best_solution):
                best_solution = copy.deepcopy(current_solution)

            # compute best solution score
            best_score = self.evaluate_solution(best_solution)
            solution_score_list.append(best_score)

            # cooling
            temperature *= self.cooling_rate

        return best_solution, self.evaluate_solution(best_solution)



