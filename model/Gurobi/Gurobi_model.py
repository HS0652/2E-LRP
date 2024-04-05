from gurobipy import *
import gurobipy as gp
import json
import math


def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def load_instance_from_file(filename):
    with open(filename, 'r') as file:
        instance = json.load(file)
    return instance


class Gurobi_Model:
    def __init__(self, **gurobi_params):
        self.depot_num = gurobi_params['depot_num']
        self.dp_num = gurobi_params['dp_num']
        self.customer_num = gurobi_params['customer_num']
        self.instance = None

        self.light_vehicle_load = 40
        self.heavy_vehicle_load = None

        self.light_vehicle_cost = gurobi_params['light_vehicle_cost']
        self.heavy_vehicle_cost = gurobi_params['heavy_vehicle_cost']
        self.depot_cost = gurobi_params['depot_cost']
        self.dp_cost = gurobi_params['dp_cost']
        self.per_distance_cost = gurobi_params['per_distance_cost']

        self.location = None
        self.customer_demand = None
        self.facility_capacity = None

    def load_instance(self, instance):
        self.instance = instance

        heavy_load_map = {
            37: 200,
            64: 150,
            120: 250,
        }
        self.heavy_vehicle_load = heavy_load_map[len(instance)]

        self.location = [[i[0], i[1]] for i in instance]
        self.customer_demand = [i[2] for i in instance]
        self.facility_capacity = [i[3] for i in instance]

    def solve_problem(self):
        model = gp.Model("gurobi_for_2E-LRP")

        # set variable
        e = model.addVars(self.depot_num + self.dp_num, self.depot_num + self.dp_num, 2 * self.depot_num,
                          vtype=GRB.BINARY, name="e")
        f = model.addVars(self.dp_num + self.customer_num, self.dp_num + self.customer_num, 2 * self.dp_num,
                          vtype=GRB.BINARY, name="f")
        h = model.addVars(self.customer_num, 2 * self.dp_num, vtype=GRB.CONTINUOUS, name="g")
        g = model.addVars(self.dp_num, 2 * self.depot_num, vtype=GRB.CONTINUOUS, name="h")
        d = model.addVars(self.dp_num, vtype=GRB.CONTINUOUS, name="d")
        u = model.addVars(self.depot_num, self.dp_num, vtype=GRB.BINARY, name="u")
        v = model.addVars(self.dp_num, self.customer_num, vtype=GRB.BINARY, name="v")
        x = model.addVars(self.depot_num, vtype=GRB.BINARY, name="x")
        y = model.addVars(self.dp_num, vtype=GRB.BINARY, name="y")

        # update variable
        model.update()

        # set objective function
        model.setObjective(gp.quicksum(x[i] * self.depot_cost
                                       for i in range(self.depot_num))
                           + gp.quicksum(y[i] * self.dp_cost
                                         for i in range(self.dp_num))
                           + gp.quicksum(e[i, j, r] * calculate_distance(self.location[i], self.location[j])
                                         for i in range(self.depot_num + self.dp_num)
                                         for j in range(self.depot_num + self.dp_num)
                                         for r in range(2 * self.depot_num))
                           + gp.quicksum(
            f[j, k, r] * calculate_distance(self.location[j + self.depot_num], self.location[k + self.depot_num])
            for j in range(self.dp_num + self.customer_num)
            for k in range(self.dp_num + self.customer_num)
            for r in range(2 * self.dp_num))
                           + gp.quicksum(e[i, j, r]
                                         for i in range(self.depot_num)
                                         for j in range(self.depot_num, self.depot_num + self.dp_num)
                                         for r in range(2 * self.depot_num)) * self.heavy_vehicle_cost
                           + gp.quicksum(f[j, k, r]
                                         for j in range(self.dp_num)
                                         for k in range(self.dp_num, self.dp_num + self.customer_num)
                                         for r in range(2 * self.dp_num)) * self.light_vehicle_cost
                           , sense=gp.GRB.MINIMIZE)

        # constraints for first echelon
        # 保证每条路径的每一个节点的出度与入度相同
        model.addConstrs((gp.quicksum(e[i, j, r] for j in range(self.depot_num + self.dp_num))
                          == gp.quicksum(e[j, i, r] for j in range(self.depot_num + self.dp_num)))
                         for i in range(self.depot_num + self.dp_num)
                         for r in range(2 * self.depot_num))
        model.addConstrs((gp.quicksum(e[i, j, r] for j in range(self.depot_num, self.depot_num + self.dp_num)) -
                          gp.quicksum(e[j, i, r] for j in range(self.depot_num, self.depot_num + self.dp_num)) == 0)
                         for i in range(self.depot_num)
                         for r in range(2 * self.depot_num))

        # 保证路径中不会包含重复的边
        model.addConstrs((e[i, j, r] + e[j, i, r] <= 1)
                         for i in range(self.depot_num + self.dp_num)
                         for j in range(self.depot_num + self.dp_num)
                         for r in range(2 * self.depot_num))

        # 确保每个节点不会自连
        model.addConstrs(
            e[i, i, r] == 0 for i in range(self.depot_num + self.dp_num) for r in range(2 * self.depot_num))

        # 确保仓库之间不会相互连接
        model.addConstrs(e[i, j, r] == 0 for i in range(self.depot_num) for j in range(self.depot_num) for r in
                         range(2 * self.depot_num))

        # 消除子回路
        model.addConstrs((2 <= g[i, r]) for i in range(self.dp_num) for r in range(2 * self.depot_num))
        model.addConstrs((g[i, r] <= self.dp_num) for i in range(self.dp_num) for r in range(2 * self.depot_num))
        model.addConstrs(
            (g[i, r] - g[j, r] + self.dp_num * e[i + self.depot_num, j + self.depot_num, r] <= self.dp_num - 1)
            for r in range(2 * self.depot_num)
            for i in range(self.dp_num)
            for j in range(self.dp_num) if i != j)

        # 确保每条路径的dp总需求不大于车辆的负载
        model.addConstrs((gp.quicksum(e[i, j, r] * d[j - self.depot_num]
                                      for i in range(self.depot_num + self.dp_num)
                                      for j in range(self.depot_num, self.depot_num + self.dp_num))
                          <= self.heavy_vehicle_load)
                         for r in range(2 * self.depot_num))

        # 确保分配的dp总需求不大于仓库容量
        model.addConstrs((gp.quicksum(u[i, j] * d[j] for j in range(self.dp_num)) <= x[i] * self.facility_capacity[i])
                         for i in range(self.depot_num))

        # 若回路存在配送点, 则必须要有一个仓库被使用
        model.addConstrs((gp.quicksum(e[i, j, r]
                                      for i in range(self.depot_num + self.dp_num)
                                      for j in range(self.depot_num + self.dp_num)) / self.dp_num
                          <= gp.quicksum(e[i, j, r]
                                         for i in range(self.depot_num)
                                         for j in range(self.depot_num + self.dp_num)))
                         for r in range(2 * self.depot_num))
        model.addConstrs((gp.quicksum(e[i, j, r] for i in range(self.depot_num)
                                      for j in range(self.depot_num, self.depot_num + self.dp_num)) <= 1)
                         for r in range(2 * self.depot_num))

        # 确定开放的仓库
        model.addConstrs((gp.quicksum(e[i, j, r] for j in range(self.depot_num, self.depot_num + self.dp_num)
                                      for r in range(2 * self.depot_num)) / self.dp_num <= x[i])
                         for i in range(self.depot_num))
        model.addConstrs((x[i] <= gp.quicksum(e[i, j, r] for j in range(self.depot_num, self.depot_num + self.dp_num)
                                              for r in range(2 * self.depot_num)))
                         for i in range(self.depot_num))

        # 确定开放的配送点
        model.addConstrs((gp.quicksum(e[i, j, r]
                                      for i in range(self.depot_num + self.dp_num)
                                      for r in range(2 * self.depot_num))
                          == y[j - self.depot_num])
                         for j in range(self.depot_num, self.depot_num + self.dp_num))
        model.addConstrs((gp.quicksum(e[i, j, r]
                                      for j in range(self.depot_num + self.dp_num)
                                      for r in range(2 * self.depot_num))
                          == y[i - self.depot_num])
                         for i in range(self.depot_num, self.depot_num + self.dp_num))
        model.addConstrs((gp.quicksum(u[i, j] for i in range(self.depot_num)) == y[j]) for j in range(self.dp_num))

        # constraints for second echelon
        # 保证每条路径节点的的出度与入度相同
        model.addConstrs((gp.quicksum(f[j, k, r] for k in range(self.dp_num + self.customer_num))
                          == gp.quicksum(f[k, j, r] for k in range(self.dp_num + self.customer_num)))
                         for j in range(self.dp_num + self.customer_num)
                         for r in range(2 * self.dp_num))

        model.addConstrs((gp.quicksum(f[j, k, r] for k in range(self.dp_num, self.dp_num + self.customer_num)) ==
                          gp.quicksum(f[k, j, r] for k in range(self.dp_num, self.dp_num + self.customer_num)))
                         for j in range(self.dp_num)
                         for r in range(2 * self.dp_num))

        # 防止节点自连
        model.addConstrs(
            f[i, i, r] == 0 for i in range(self.dp_num + self.customer_num) for r in range(2 * self.dp_num))

        # 防止dp之间相互连接
        model.addConstrs((f[j, k, r] == 0)
                         for j in range(self.dp_num)
                         for k in range(self.dp_num)
                         for r in range(2 * self.dp_num))

        # 保证路径中不会包含重复的边
        model.addConstrs((f[j, k, r] + f[k, j, r] <= 1)
                         for j in range(self.dp_num + self.customer_num)
                         for k in range(self.dp_num + self.customer_num)
                         for r in range(2 * self.dp_num))

        # 消除子回路
        model.addConstrs((2 <= h[i, r]) for i in range(self.customer_num) for r in range(2 * self.dp_num))
        model.addConstrs(
            (h[i, r] <= self.customer_num) for i in range(self.customer_num) for r in range(2 * self.dp_num))
        model.addConstrs(
            (h[i, r] - h[j, r] + self.customer_num * f[i + self.dp_num, j + self.dp_num, r] <= self.customer_num - 1)
            for r in range(2 * self.dp_num)
            for i in range(self.customer_num)
            for j in range(self.customer_num) if i != j)

        # 确定每个开放的dp的总需求
        model.addConstrs((gp.quicksum(v[j, k] * self.customer_demand[k + self.depot_num + self.dp_num]
                                      for k in range(self.customer_num))) == d[j]
                         for j in range(self.dp_num))

        # 确保每条路径的客户总需求不大于车辆负载
        model.addConstrs((gp.quicksum(f[j, k, r] * self.customer_demand[k + self.depot_num]
                                      for j in range(self.dp_num + self.customer_num)
                                      for k in range(self.dp_num, self.dp_num + self.customer_num))
                          <= self.light_vehicle_load)
                         for r in range(2 * self.dp_num))

        # 确保安排给仓库的客户总需求不大于仓库容量
        model.addConstrs((gp.quicksum(v[j, k] * self.customer_demand[k + self.depot_num + self.dp_num]
                                      for k in range(self.customer_num))
                          <= y[j] * self.facility_capacity[j + self.depot_num])
                         for j in range(self.dp_num))

        # 确保每个客户只被服务一次
        model.addConstrs((gp.quicksum(
            f[j, k, r] for j in range(self.dp_num + self.customer_num) for r in range(2 * self.dp_num)) == 1)
                         for k in range(self.dp_num, self.dp_num + self.customer_num))
        model.addConstrs((gp.quicksum(
            f[j, k, t] for k in range(self.dp_num + self.customer_num) for t in range(2 * self.dp_num)) == 1)
                         for j in range(self.dp_num, self.dp_num + self.customer_num))
        model.addConstrs((gp.quicksum(v[j, k] for j in range(self.dp_num)) == 1) for k in range(self.customer_num))

        # 将配送点的开放与否
        model.addConstrs((gp.quicksum(f[j, k, r] for k in range(self.dp_num, self.dp_num + self.customer_num)
                                      for r in range(2 * self.dp_num)) / self.customer_num <= y[j])
                         for j in range(self.dp_num))
        model.addConstrs((y[j] <= gp.quicksum(f[j, k, r] for k in range(self.dp_num, self.dp_num + self.customer_num)
                                              for r in range(2 * self.dp_num)))
                         for j in range(self.dp_num))

        # 若路径中存在客户，则必须有一个配送点
        model.addConstrs((gp.quicksum(f[j, k, r] for j in range(self.dp_num + self.customer_num)
                                      for k in range(self.dp_num + self.customer_num)) / self.customer_num
                          <= gp.quicksum(f[j, k, r] for j in range(self.dp_num)
                                         for k in range(self.dp_num + self.customer_num)))
                         for r in range(2 * self.dp_num))
        model.addConstrs((gp.quicksum(
            f[j, k, r] for j in range(self.dp_num) for k in range(self.dp_num, self.dp_num + self.customer_num)) <= 1)
                         for r in range(2 * self.dp_num))

        # set time limit to 3600s
        model.setParam("TimeLimit", 10000)

        # solve the problem
        model.optimize()

        # print the result
        if model.status == gp.GRB.OPTIMAL:
            print("Optimal solution found!")
            print("Objective value:", model.objVal)
            print("Time: ", model.Runtime)
            return model.objVal, model.Runtime
        elif model.status == gp.GRB.INFEASIBLE:
            print("Model is infeasible")
        elif model.status == gp.GRB.UNBOUNDED:
            print("Model is unbounded")
        else:
            print("Optimization ended with status:", model.status)
            print("Objective value:", model.objVal)
            print("Time: ", model.Runtime)
            return model.objVal, model.Runtime
