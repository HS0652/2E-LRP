from dataclasses import dataclass
import torch
import numpy as np


@dataclass
class Reset_State:
    depot_node: torch.Tensor = None
    dp_node: torch.Tensor = None
    customer_node: torch.Tensor = None


@dataclass
class Step_State:
    Batch_Idx: torch.Tensor = None
    Lrp_Idx: torch.Tensor = None
    # shape: (batch, lrp)
    selected_count: int = None
    current_node: torch.Tensor = None
    # shape: (batch, lrp)
    total_mask: torch.Tensor = None
    # shape: (batch, lrp, node_num)
    finished: torch.Tensor = None
    # shape: (batch, lrp)
    remaining_capacity: torch.Tensor = None
    remaining_load: torch.Tensor = None
    # shape: (batch, lrp)


class LRP_Env:
    def __init__(self, **env_params):
        # Const @Init
        self.env_params = env_params
        self.depot_num = env_params['depot_num']
        self.dp_num = env_params['distribution_point_num']
        self.customer_num = env_params['customer_num']
        self.node_num = self.depot_num + self.dp_num + self.customer_num

        # Const @load_Problem
        self.lrp_size = self.dp_num
        self.batch_size = None

        self.per_depot_capacity = None
        self.per_dp_capacity = None
        self.per_heavy_vehicle_load = None
        self.per_light_vehicle_load = None

        self.per_depot_cost = None
        self.per_dp_cost = None
        self.per_heavy_vehicle_cost = None
        self.per_light_vehicle_cost = None
        self.per_distances_cost = None

        self.depot_capacity = None
        self.depot_cost = None
        self.dp_capacity = None
        self.dp_cost = None
        self.customer_demand = None
        self.vehicle_load = None
        self.node_loc = None
        self.node_list = None

        self.Batch_Idx: torch.Tensor = None
        self.Lrp_Idx: torch.Tensor = None

        # Dynamic_1
        self.init_dp_demand = None
        self.customer_demand_list = None
        self.dp_demand_list = None
        self.remaining_load = None
        self.remaining_capacity = None
        self.selected_count = None
        self.previous_node = None
        self.previous_dp = None
        self.previous_depot = None
        self.current_node = None
        self.current_depot = None
        self.current_dp = None
        self.selected_node_list = None
        self.acc_reward = None

        # Dynamic_2
        self.at_the_depot = None
        self.at_the_dp = None
        self.visited_flag = None
        self.depot_mask = None
        self.dp_mask = None
        self.customer_mask = None
        self.total_mask = None
        self.finished = None
        self.vehicle_cost = 0
        self.facility_cost = 0

        # State
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def use_saved_problems(self, filename, device):
        self.FLAG__use_saved_problems = True

        loaded_dict = torch.load(filename, map_location=device)
        self.saved_depot_xy = loaded_dict['depot_xy']
        self.saved_node_xy = loaded_dict['node_xy']
        self.saved_node_demand = loaded_dict['node_demand']
        self.saved_index = 0

    def load_instance(self, instance: list):
        self.batch_size = 1

        self.per_depot_cost = 500
        self.per_dp_cost = 200
        self.per_heavy_vehicle_cost = 50
        self.per_light_vehicle_cost = 20
        self.per_distances_cost = 1

        self.per_heavy_vehicle_load = 300
        self.per_light_vehicle_load = 50
        self.per_dp_capacity = [instance[self.depot_num][3]]
        self.per_depot_capacity = [instance[0][3]]

        self.customer_demand = torch.tensor([node[2] for node in instance]).unsqueeze(0)
        facility_capacity = torch.tensor([node[3] for node in instance])
        self.depot_capacity = torch.zeros(size=(self.batch_size, self.node_num), dtype=torch.int32)
        self.depot_capacity[:, :self.depot_num] = facility_capacity[:self.depot_num]
        self.dp_capacity = torch.zeros(size=(self.batch_size, self.node_num), dtype=torch.int32)
        self.dp_capacity[:, self.depot_num:self.depot_num + self.dp_num] = \
            facility_capacity[self.depot_num:self.depot_num + self.dp_num]

        self.vehicle_load = torch.zeros(size=(self.batch_size, self.node_num))
        self.vehicle_load[:, :self.depot_num] = self.per_heavy_vehicle_load
        self.vehicle_load[:, self.depot_num:self.depot_num + self.dp_num] = self.per_light_vehicle_load

        self.depot_cost = torch.zeros(size=(self.batch_size, self.node_num))
        self.depot_cost[:, :self.depot_num] = self.per_depot_cost
        self.dp_cost = torch.zeros(size=(self.batch_size, self.node_num))
        self.dp_cost[:, self.depot_num: self.depot_num + self.dp_num] = self.per_dp_cost

        node_x = torch.tensor([node[0] for node in instance]).unsqueeze(0).unsqueeze(2)
        node_y = torch.tensor([node[1] for node in instance]).unsqueeze(0).unsqueeze(2)
        self.node_loc = torch.cat((node_x, node_y), dim=2)

        self.node_list = torch.cat((node_x, node_y, self.customer_demand.unsqueeze(2), self.depot_capacity.unsqueeze(2),
                                    self.depot_cost.unsqueeze(2), self.dp_capacity.unsqueeze(2),
                                    self.dp_cost.unsqueeze(2), self.vehicle_load.unsqueeze(2)), dim=2)
        # shape: (batch, node_num, 8)

        self.Batch_Idx = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.lrp_size)
        self.Lrp_Idx = torch.arange(self.lrp_size)[None, :].expand(self.batch_size, self.lrp_size)
        self.step_state.Batch_Idx = self.Batch_Idx
        self.step_state.Lrp_Idx = self.Lrp_Idx

        self.reset_state.depot_node = self.node_list[:, :self.depot_num, :]
        self.reset_state.dp_node = self.node_list[:, self.depot_num: self.depot_num+self.dp_num, :]
        self.reset_state.customer_node = self.node_list[:, self.depot_num+self.dp_num:, :]

    def load_problems(self, batch_size):
        self.batch_size = batch_size

        self.per_depot_cost = 500
        self.per_dp_cost = 200
        self.per_heavy_vehicle_cost = 50
        self.per_light_vehicle_cost = 20
        self.per_distances_cost = 1

        self.customer_demand = torch.randint(1, 10, size=(batch_size, self.node_num))
        self.customer_demand[:, :self.depot_num+self.dp_num] = 0

        total_customer_demand = torch.sum(self.customer_demand, dim=1, keepdim=True)
        self.per_depot_capacity = torch.div(total_customer_demand, 3, rounding_mode='trunc') + 50
        self.per_dp_capacity = torch.div(total_customer_demand, 10, rounding_mode='trunc') + 3
        self.per_heavy_vehicle_load = 300.00
        self.per_light_vehicle_load = 40.00

        self.depot_cost = torch.zeros(size=(batch_size, self.node_num))
        self.depot_cost[:, :self.depot_num] = self.per_depot_cost
        self.depot_capacity = torch.zeros(size=(batch_size, self.node_num))
        self.depot_capacity[:, :self.depot_num] = self.per_depot_capacity

        self.dp_capacity = torch.zeros(size=(batch_size, self.node_num))
        self.dp_capacity[:, self.depot_num:self.depot_num+self.dp_num] = self.per_dp_capacity
        self.dp_cost = torch.zeros(size=(batch_size, self.node_num))
        self.dp_cost[:, self.depot_num: self.depot_num+self.dp_num] = self.per_dp_cost

        self.vehicle_load = torch.zeros(size=(batch_size, self.node_num))
        self.vehicle_load[:, :self.depot_num] = self.per_heavy_vehicle_load
        self.vehicle_load[:, self.depot_num:self.depot_num+self.dp_num] = self.per_light_vehicle_load

        depot_loc = torch.randint(0, 100, size=(batch_size, self.depot_num, 2))
        dp_loc = torch.randint(1, 100, size=(batch_size, self.dp_num, 2))
        customer_loc = torch.randint(0, 100, size=(batch_size, self.customer_num, 2))
        self.node_loc = torch.cat((depot_loc, dp_loc, customer_loc), dim=1)
        node_x = self.node_loc[:, :, 0].unsqueeze(2)
        node_y = self.node_loc[:, :, 1].unsqueeze(2)
        self.node_list = torch.cat((node_x, node_y, self.customer_demand.unsqueeze(2), self.depot_capacity.unsqueeze(2),
                                    self.depot_cost.unsqueeze(2), self.dp_capacity.unsqueeze(2),
                                    self.dp_cost.unsqueeze(2), self.vehicle_load.unsqueeze(2)), dim=2)
        # shape: (batch, node_num, 8)

        self.Batch_Idx = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.lrp_size)
        self.Lrp_Idx = torch.arange(self.lrp_size)[None, :].expand(self.batch_size, self.lrp_size)
        self.step_state.Batch_Idx = self.Batch_Idx
        self.step_state.Lrp_Idx = self.Lrp_Idx

        self.reset_state.depot_node = self.node_list[:, :self.depot_num, :]
        self.reset_state.dp_node = self.node_list[:, self.depot_num: self.depot_num+self.dp_num, :]
        self.reset_state.customer_node = self.node_list[:, self.depot_num+self.dp_num:, :]

    def reset(self):
        self.selected_count = 0
        self.previous_node = None
        self.current_node = None
        self.current_dp = torch.zeros(size=(self.batch_size, self.lrp_size))

        self.selected_node_list = torch.zeros((self.batch_size, self.lrp_size, 0), dtype=torch.long)

        self.remaining_load = torch.zeros(size=(self.batch_size, self.lrp_size))
        self.remaining_load += self.per_light_vehicle_load
        self.remaining_capacity = torch.zeros(size=(self.batch_size, self.lrp_size))
        self.remaining_capacity += self.per_dp_capacity[0]

        self.acc_reward = torch.zeros(size=(self.batch_size, self.lrp_size))

        self.at_the_depot = torch.zeros(size=(self.batch_size, self.lrp_size), dtype=torch.bool)
        self.at_the_dp = torch.ones(size=(self.batch_size, self.lrp_size), dtype=torch.bool)

        self.depot_mask = torch.zeros(size=(self.batch_size, self.lrp_size, self.node_num))
        self.depot_mask[:, :, :self.depot_num] = float('-inf')
        self.dp_mask = torch.zeros(size=(self.batch_size, self.lrp_size, self.node_num))
        self.dp_mask[:, :, self.depot_num:self.dp_num+self.depot_num] = float('-inf')
        self.customer_mask = torch.zeros(size=(self.batch_size, self.lrp_size, self.node_num))
        self.customer_mask[:, :, self.dp_num+self.depot_num:] = float('-inf')
        self.total_mask = torch.zeros(size=(self.batch_size, self.lrp_size, self.node_num))
        self.total_mask = self.total_mask + self.depot_mask + self.customer_mask

        self.visited_flag = torch.zeros(size=(self.batch_size, self.lrp_size, self.node_num))

        self.customer_demand_list = self.customer_demand[:, None, :].expand(self.batch_size, self.lrp_size, -1)
        self.dp_demand_list = torch.zeros(size=(self.batch_size, self.lrp_size, self.node_num), dtype=torch.int)

        self.finished = torch.zeros(size=(self.batch_size, self.lrp_size), dtype=torch.bool)

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.remaining_load = self.remaining_load
        self.step_state.remaining_capacity = self.remaining_capacity
        self.step_state.current_node = self.current_node
        self.step_state.total_mask = self.total_mask
        self.step_state.finished = self.finished

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # base
        round_error_epsilon = 0.00001

        self.selected_count += 1
        gather_index = selected[:, :, None]
        self.current_node = selected
        if self.selected_count == 1:
            self.current_dp = selected
        self.selected_node_list = torch.cat((self.selected_node_list, selected[:, :, None]), dim=2)
        self.visited_flag[self.Batch_Idx, self.Lrp_Idx, selected] = float('-inf')
        self.total_mask = self.visited_flag.clone()

        # phase_1: Goods are dispatched from distribution points to customers
        if self.selected_count <= (self.dp_num + self.customer_num):
            self.total_mask += self.depot_mask

            selected_demand = self.customer_demand_list.gather(dim=2, index=gather_index).squeeze(dim=2)
            mask = torch.ones_like(self.customer_demand_list)
            src_zero = torch.zeros_like(gather_index)
            mask.scatter_(dim=2, index=gather_index, src=src_zero)
            self.customer_demand_list = self.customer_demand_list * mask

            self.at_the_dp = torch.Tensor(selected < (self.depot_num + self.dp_num))
            dp_index = torch.nonzero(self.at_the_dp, as_tuple=False)
            self.current_dp = torch.where(self.at_the_dp, selected, self.current_dp)

            self.remaining_capacity -= selected_demand
            self.remaining_load -= selected_demand
            demand_over_load = torch.nonzero(self.remaining_load < 0)

            for i, j in dp_index:
                self.remaining_capacity[i, j] = self.per_dp_capacity[i]
                self.remaining_load[i, j] = self.per_light_vehicle_load

            for i, j in demand_over_load:
                self.remaining_load[i, j] = self.per_light_vehicle_load - selected_demand[i, j]

            demand_over_capacity = self.remaining_capacity[:, :, None] + round_error_epsilon < self.customer_demand_list

            dp_demand = torch.zeros_like(self.dp_demand_list)
            dp_demand = dp_demand.to(selected_demand.dtype)
            dp_demand = dp_demand.scatter_(2, self.current_dp[:, :, None], selected_demand[:, :, None])
            self.dp_demand_list += dp_demand

            cur_customer_demand = self.customer_demand_list.clone()
            cur_customer_demand[demand_over_capacity] = 0
            exist_customer_to_choose = ~((cur_customer_demand == 0).all(2))
            index_mask_dp = torch.nonzero(exist_customer_to_choose)
            for i, j in index_mask_dp:
                self.total_mask[i, j, :self.depot_num+self.dp_num] = float('-inf')
            index_to_demask_dp = torch.nonzero((self.total_mask == float('-inf')).all(2))
            for i, j in index_to_demask_dp:
                self.total_mask[i, j, self.depot_num:self.depot_num+self.dp_num] = 0

            self.total_mask[demand_over_capacity] = float('-inf')
            self.total_mask += self.visited_flag

            self.previous_node = self.current_node
            self.previous_dp = self.current_dp

            if self.selected_count == (self.dp_num + self.customer_num):
                self.total_mask[:, :, :self.depot_num] = 0
                self.total_mask += self.dp_mask
                self.total_mask += self.customer_mask
                self.visited_flag[:, :, self.depot_num:self.depot_num + self.dp_num] = 0

        # pre_phase_2: reset state and choose a depot
        if self.selected_count == (self.dp_num + self.customer_num + 1):
            self.total_mask += self.depot_mask

            self.remaining_load = torch.zeros(size=(self.batch_size, self.lrp_size))
            self.remaining_load += self.per_heavy_vehicle_load
            self.remaining_capacity = torch.zeros(size=(self.batch_size, self.lrp_size))
            self.remaining_capacity += self.per_depot_capacity[0]

            self.current_depot = selected
            self.previous_node = selected
            self.init_dp_demand = self.dp_demand_list

            dp_zero_demand = ~(self.init_dp_demand.to(torch.bool))
            dp_zero_demand[:, :, :self.depot_num] = False
            dp_zero_demand[:, :, self.depot_num + self.dp_num:] = False

            self.total_mask[dp_zero_demand] = float('-inf')

        # phase_2: Goods are dispatched from depots to distribution points
        if self.selected_count > (self.dp_num + self.customer_num + 1):
            selected_demand = self.dp_demand_list.gather(dim=2, index=gather_index).squeeze(dim=2)
            mask = torch.ones_like(self.dp_demand_list)
            src_zero = torch.zeros_like(gather_index).to(mask.dtype)
            mask.scatter_(dim=2, index=gather_index, src=src_zero)
            self.dp_demand_list = self.dp_demand_list * mask

            self.at_the_depot = torch.Tensor(selected < self.depot_num)
            depot_index = torch.nonzero(self.at_the_depot)
            self.current_depot = torch.where(self.at_the_depot, selected, self.current_depot)

            self.remaining_capacity -= selected_demand
            self.remaining_load -= selected_demand
            demand_over_load = torch.nonzero(self.remaining_load < 0)

            for i, j in depot_index:
                self.remaining_capacity[i, j] = self.per_depot_capacity[i]
                self.remaining_load[i, j] = self.per_heavy_vehicle_load

            for i, j in demand_over_load:
                self.remaining_load[i, j] = self.per_heavy_vehicle_load - selected_demand[i, j]

            demand_over_capacity = self.dp_demand_list > self.remaining_capacity[:, :, None]

            self.total_mask[demand_over_capacity] = float('-inf')

            dp_zero_demand = ~(self.init_dp_demand.to(torch.bool))
            dp_zero_demand[:, :, :self.depot_num] = False
            dp_zero_demand[:, :, self.depot_num+self.dp_num:] = False

            zero_demand_index = torch.nonzero(dp_zero_demand)
            ignore_distance_index = []
            for i, j, k in zero_demand_index:
                if selected[i, j] == k:
                    ignore_distance_index.append([i, j])

            cur_dp_demand = self.dp_demand_list.clone()
            cur_dp_demand[demand_over_capacity] = 0
            exist_dp_to_choose = ~((cur_dp_demand == 0).all(2))
            index_mask_depot = torch.nonzero(exist_dp_to_choose)
            for i, j in index_mask_depot:
                self.total_mask[i, j, :self.depot_num] = float('-inf')

            index_to_demask_depot = torch.nonzero((self.total_mask == float('-inf')).all(2))
            for i, j in index_to_demask_depot:
                self.total_mask[i, j, :self.depot_num] = 0
            self.total_mask[dp_zero_demand] = float('-inf')
            self.total_mask += self.visited_flag

            index_to_demask_dp = torch.nonzero((self.total_mask == float('-inf')).all(2))
            for i, j in index_to_demask_dp:
                self.total_mask[i, j, :self.depot_num+self.dp_num] = 0
            self.total_mask += self.visited_flag

            self.previous_node = self.current_node
            self.previous_depot = self.current_depot

        newly_finished = (self.visited_flag == float('-inf')).all(dim=2)
        self.finished = self.finished + newly_finished

        self.step_state.selected_count = self.selected_count
        self.step_state.remaining_load = self.remaining_load
        self.step_state.remaining_capacity = self.remaining_capacity
        self.step_state.current_node = self.current_node
        self.step_state.total_mask = self.total_mask
        self.step_state.finished = self.finished

        done = self.finished.all()
        selected_node = []
        if done:
            reward, selected_node = self._get_total_reward()
        else:
            reward = None

        return self.step_state, reward, done, selected_node

    def _get_total_reward(self):
        route1 = [[[] for _ in range(self.lrp_size)] for _ in range(self.batch_size)]
        route2 = [[[] for _ in range(self.lrp_size)] for _ in range(self.batch_size)]
        choose_dp = [[[] for _ in range(self.lrp_size)] for _ in range(self.batch_size)]
        choose_depot = [[[] for _ in range(self.lrp_size)] for _ in range(self.batch_size)]
        total_reward = [[[] for _ in range(self.lrp_size)] for _ in range(self.batch_size)]
        for batch in range(self.batch_size):
            for lrp in range(self.lrp_size):
                i = 1
                node_list = self.node_list.clone()
                selected_node_list = self.selected_node_list[batch][lrp]
                cur_dp = pre_index = selected_node_list[0]
                remaining_load = self.per_light_vehicle_load
                route1[batch][lrp].append(pre_index)
                choose_dp[batch][lrp].append(pre_index)
                acc_demand = 0
                for _ in range(1, self.dp_num + self.customer_num):
                    cur_index = selected_node_list[i]
                    cur_node = node_list[batch][cur_index]
                    cur_demand = cur_node[2]
                    if pre_index < self.depot_num + self.dp_num and cur_index < self.depot_num + self.dp_num:
                        node_list[batch][cur_index][5] = 0
                        node_list[batch][pre_index][5] = 0
                        pre_index = cur_index
                        i += 1
                        continue
                    if cur_index < self.depot_num + self.dp_num:
                        remaining_load = self.per_light_vehicle_load
                        route1[batch][lrp].append(cur_dp)
                        node_list[batch][cur_dp][5] = acc_demand
                        choose_dp[batch][lrp].append(cur_index)
                        cur_dp = cur_index
                        acc_demand = 0
                    else:
                        remaining_load -= cur_demand
                        acc_demand += cur_demand
                        if remaining_load < 0:
                            route1[batch][lrp].append(cur_dp)
                            remaining_load = self.per_light_vehicle_load - cur_demand
                    route1[batch][lrp].append(cur_index)
                    pre_index = cur_index
                    i += 1
                route1[batch][lrp].pop()
                choose_dp[batch][lrp].pop()
                cur_depot = pre_index = selected_node_list[i]
                remaining_load = self.per_heavy_vehicle_load
                route2[batch][lrp].append(pre_index)
                i += 1
                for _ in range(1, self.depot_num + self.dp_num):
                    cur_index = selected_node_list[i]
                    cur_node = node_list[batch][cur_index]
                    cur_demand = cur_node[5]
                    if pre_index < self.depot_num and cur_node[5] == 0:
                        break
                    if pre_index < self.depot_num and cur_index < self.depot_num:
                        break
                    if cur_index < self.depot_num:
                        remaining_load = self.per_heavy_vehicle_load
                        route2[batch][lrp].append(cur_depot)
                        choose_depot[batch][lrp].append(cur_depot)
                        cur_depot = cur_index
                    else:
                        remaining_load -= cur_demand
                        if remaining_load < 0:
                            route2[batch][lrp].append(cur_depot)
                            remaining_load = self.per_heavy_vehicle_load - cur_demand
                    route2[batch][lrp].append(cur_index)
                    pre_index = cur_index
                    i += 1
                route2[batch][lrp].pop()

                depot_num = len(choose_depot[batch][lrp])
                dp_num = len(choose_dp[batch][lrp])
                facility_cost = depot_num * self.per_depot_cost + dp_num * self.per_dp_cost

                light_vehicle_num = sum(route1[batch][lrp].count(dp) for dp in choose_dp[batch][lrp]) - dp_num
                heavy_vehicle_num = sum(route2[batch][lrp].count(depot) for depot in choose_depot[batch][lrp])-depot_num
                vehicle_cost = light_vehicle_num * self.per_light_vehicle_cost \
                               + heavy_vehicle_num * self.per_heavy_vehicle_cost

                route1_distance = 0
                pre_node = node_list[batch][route1[batch][lrp][0]]
                for i in range(1, len(route1[batch][lrp])):
                    cur_node = node_list[batch][route1[batch][lrp][i]]
                    if pre_node[2] == 0 and cur_node[2] == 0:
                        route1_distance += 0  # if previous node and currant node are dp, the distance is zero.
                    else:
                        route1_distance += ((pre_node[:2] - cur_node[:2]) ** 2).sum().sqrt()

                    pre_node = cur_node

                route2_distance = 0
                pre_node = node_list[batch][route2[batch][lrp][0]]
                for i in range(1, len(route2[batch][lrp])):
                    cur_node = node_list[batch][route2[batch][lrp][i]]
                    if pre_node[5] == 0 and cur_node[5] == 0:
                        route2_distance += 0  # if previous node and currant node is depot, the distance is zero.
                    else:
                        route2_distance += ((pre_node[:2] - cur_node[:2]) ** 2).sum().sqrt()
                    pre_node = cur_node

                distance_cost = (route1_distance + route2_distance) * self.per_distances_cost
                total_cost = -(facility_cost + vehicle_cost + distance_cost)
                total_reward[batch][lrp].append(total_cost)
        total_reward = torch.tensor(total_reward).squeeze(2)

        # reward_list = [i.item() for i in total_reward[0]]
        # index = reward_list.index(max(reward_list))
        # print('score: {}'.format(-reward_list[index]))
        # route1_list = [i.item() for i in route1[0][index]]
        # route2_list = [i.item() for i in route2[0][index]]
        # print('route1: {}'.format(route1_list))
        # print('route2: {}'.format(route2_list))
        reward_list = [i.item() for i in total_reward[0]]
        index = reward_list.index(max(reward_list))

        return total_reward, self.selected_node_list[0][index]










