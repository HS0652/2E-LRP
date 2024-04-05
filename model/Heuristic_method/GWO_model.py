import numpy as np
import random
import copy

def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def sort_points_by_distance(points):
    sorted_points = [points[0]]  # Start with the first point as the initial point

    while len(sorted_points) < len(points):
        current_point = sorted_points[-1]
        remaining_points = [point for point in points if point not in sorted_points]
        remaining_points.sort(key=lambda x: euclidean_distance(current_point, x))
        sorted_points.append(remaining_points[0])

    return sorted_points

class GWO_MODEL:
    def __init__(self, **gwo_params):
        # Define the parameters for problem
        self.depot_num = gwo_params['depot_num']
        self.dp_num = gwo_params['dp_num']
        self.customer_num = gwo_params['customer_num']
        self.node_list = None
        self.node_num = None

        self.light_vehicle_load = 50
        self.heavy_vehicle_load = None

        self.light_vehicle_cost = gwo_params['light_vehicle_cost']
        self.heavy_vehicle_cost = gwo_params['heavy_vehicle_cost']
        self.depot_cost = gwo_params['depot_cost']
        self.dp_cost = gwo_params['dp_cost']
        self.per_distance_cost = gwo_params['per_distance_cost']

        # Define the parameters for GWO
        self.num_wolves = gwo_params['num_wolves']
        self.max_iter = gwo_params['max_iter']
        self.a = gwo_params['a']

    def load_node_list(self, instance):
        self.node_list = instance
        self.node_num = len(instance)

        heavy_load_map = {
            37: 200,
            64: 150,
            120: 250,
            230: 300,
            340: 350,
        }
        self.light_vehicle_load = 40
        self.heavy_vehicle_load = heavy_load_map[self.node_num]

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

    def initial_alpha(self):

        node_list = copy.deepcopy(self.node_list)
        depot_list = node_list[0: self.depot_num]
        dp_list = node_list[self.depot_num: self.depot_num + self.dp_num]
        customer_list = node_list[self.depot_num + self.dp_num:]

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
        alpha = [customer_index, dp_index, depot_index, route_1, route_2, selected_dp, selected_depot]

        return alpha

    def generate_wolf(self):
        customer_index = random.sample(range(self.node_num - self.customer_num, self.node_num), self.customer_num)
        node_list = copy.deepcopy(self.node_list)
        depot_list = node_list[0: self.depot_num]
        dp_list = node_list[self.depot_num: self.depot_num + self.dp_num]
        customer_list = node_list[self.depot_num + self.dp_num:]

        dp_cc = [sum(euclidean_distance(dp, i) for i in customer_list) for dp in dp_list]
        sorted_indices = sorted(enumerate(dp_cc, start=self.depot_num), key=lambda x: x[1])
        dp_index = [index for index, value in sorted_indices]
        sort_dp_list = [node_list[i] for i in dp_index]

        depot_cc = [sum(euclidean_distance(depot, i) for i in sort_dp_list) for depot in depot_list]
        sorted_depot_indices = sorted(enumerate(depot_cc), key=lambda x: x[1])
        depot_index = [index for index, value in sorted_depot_indices]

        route_1, route_2, selected_dp, selected_depot = self.generate_route(customer_index, dp_index, depot_index)
        wolf = [customer_index, dp_index, depot_index, route_1, route_2, selected_dp, selected_depot]

        return wolf

    def reconstruct_wolf(self, customer_index, dp_index, depot_index):
        seen_values = set()

        new_depot_index = []
        for x in depot_index:
            if x >= self.depot_num:
                x = x % self.depot_num
            while x in seen_values:
                x = x + 1
                if x >= self.depot_num:
                    x = 0
            new_depot_index.append(x)
            seen_values.add(x)

        new_dp_index = []
        for x in dp_index:
            if x < self.depot_num or x >= self.depot_num+self.dp_num:
                x = (x - self.depot_num) % self.dp_num + self.depot_num
            while x in seen_values:
                x = x + 1
                if x >= self.depot_num + self.dp_num:
                    x = self.depot_num
            new_dp_index.append(x)
            seen_values.add(x)

        new_customer_index = []
        for x in customer_index:
            if x < self.depot_num + self.dp_num or x >= self.node_num:
                x = (x - self.depot_num + self.dp_num) % self.customer_num + self.depot_num + self.dp_num
            while x in seen_values:
                x = x + 1
                if x >= self.node_num:
                    x = self.depot_num + self.dp_num
            new_customer_index.append(x)
            seen_values.add(x)

        route_1, route_2, selected_dp, selected_depot = self.generate_route(new_customer_index, new_dp_index, new_depot_index)
        wolf = [new_customer_index, new_dp_index, new_depot_index, route_1, route_2, selected_dp, selected_depot]

        return wolf

    def evaluate_fitness(self, wolf):
        route_1 = wolf[3]
        route_2 = wolf[4]
        selected_dp = wolf[5]
        selected_depot = wolf[6]
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

    def update_position(self, wolf, alpha, beta, delta, a):

        for i in range(3):
            for j in range(len(wolf[i])):
                A1 = 2 * a * random.random() - a
                C1 = 2 * random.random()
                D_alpha = abs(C1 * alpha[i][j] - wolf[i][j])
                X1 = alpha[i][j] - A1 * D_alpha

                A2 = 2 * a * random.random() - a
                C2 = 2 * random.random()
                D_beta = abs(C2 * beta[i][j] - wolf[i][j])
                X2 = beta[i][j] - A2 * D_beta

                A3 = 2 * a * random.random() - a
                C3 = 2 * random.random()
                D_delta = abs(C3 * delta[i][j] - wolf[i][j])
                X3 = delta[i][j] - A3 * D_delta

                wolf[i][j] = abs(int((X1 + X2 + X3) // 3))

        wolf = self.reconstruct_wolf(wolf[0], wolf[1], wolf[2])
        return wolf

    def simulated_predation_behavior(self, instance):
        # initial node list
        self.load_node_list(instance)

        wolves = [self.generate_wolf() for _ in range(self.num_wolves)]
        sorted_wolves = sorted(wolves, key=lambda x: self.evaluate_fitness(x))
        alpha = self.initial_alpha()
        beta, delta = sorted_wolves[:2]

        for iteration in range(self.max_iter):
            # Update alpha, beta, and delta wolves
            sorted_wolves = sorted(wolves, key=lambda x: self.evaluate_fitness(x))
            alpha_score = self.evaluate_fitness(alpha)
            beta_score = self.evaluate_fitness(beta)
            delta_score = self.evaluate_fitness(delta)
            for wolf in sorted_wolves:
                fitness = self.evaluate_fitness(wolf)
                if fitness < alpha_score:
                    alpha = wolf
                if alpha_score < fitness < beta_score:
                    beta = wolf
                if beta_score < fitness < delta_score:
                    delta = wolf

            a = self.a - self.a * iteration / self.max_iter

            # Update the position of each wolf
            for i in range(self.num_wolves):
                wolves[i] = self.update_position(wolves[i], alpha, beta, delta, a)

        wolves.append(alpha)
        wolves.append(beta)
        wolves.append(delta)

        best_solution = min(wolves, key=lambda x: self.evaluate_fitness(x))
        best_fitness = self.evaluate_fitness(best_solution)

        # print(f"best_fitness: {best_fitness}")

        return best_solution, best_fitness


