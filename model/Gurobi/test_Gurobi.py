import time
from datetime import datetime
import json
import logging
from Gurobi_model import Gurobi_Model


def load_instance_from_file(filename):
    with open(filename, 'r') as file:
        instances = json.load(file)
    return instances


gurobi_param = {
    'depot_num': 2,
    'dp_num': 5,
    'customer_num': 30,

    'light_vehicle_cost': 20,
    'heavy_vehicle_cost': 50,
    'depot_cost': 500,
    'dp_cost': 200,
    'per_distance_cost': 1,
}

# define logger
Formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

Gurobi_log_1 = logging.getLogger('Gurobi_for_n30')
Gurobi_log_1.setLevel(logging.INFO)
Gurobi_handler_1 = logging.FileHandler('result/log/Gurobi_for_n30.log')
Gurobi_handler_1.setFormatter(Formatter)
Gurobi_log_1.addHandler(Gurobi_handler_1)

Gurobi_log_2 = logging.getLogger('Gurobi_for_n50')
Gurobi_log_2.setLevel(logging.INFO)
Gurobi_handler_2 = logging.FileHandler('result/log/Gurobi_for_n50_2.log')
Gurobi_handler_2.setFormatter(Formatter)
Gurobi_log_2.addHandler(Gurobi_handler_2)

# test Gurobi for n30
print("Now test Gurobi for n30. \n")
Gurobi_log_1.info(f"test Gurobi with instance_2_5_30\n")
instances_1 = load_instance_from_file('instance/instance_2_5_30.json')
solution_score = []
solve_time = []
for i in range(1):
    gurobi_model = Gurobi_Model(**gurobi_param)
    gurobi_model.load_instance(instances_1[i])
    obj, time1 = gurobi_model.solve_problem()

    solution_score.append(obj)
    solve_time.append(time1)
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print(formatted_time)
    print(f"instance_{i} have done. \n")

    Gurobi_log_1.info(f"solving {i} instance take {time1}s.")
    Gurobi_log_1.info(f"{i} instance score: {obj}.")

average_time = sum(solve_time) / 1
average_score = sum(solution_score) / 1

Gurobi_log_1.info(f"----------------------------------------------------------")
Gurobi_log_1.info(f"average solving time is {average_time}s.")
Gurobi_log_1.info(f"average solution score is {average_score}.")
#
# gurobi_param['depot_num'] = 4
# gurobi_param['dp_num'] = 10
# gurobi_param['customer_num'] = 50
#
# print("Now test Gurobi for n50.\n")
# Gurobi_log_2.info(f"test Gurobi with instance_4_10_50.\n")
# instances_2 = load_instance_from_file('instance/instance_4_10_50.json')
# instances_num = len(instances_2)
# solution_score = []
# solve_time = []
# for i in range(64, 100):
#     gurobi_model = Gurobi_Model(**gurobi_param)
#     gurobi_model.load_instance(instances_2[i])
#     obj, time2 = gurobi_model.solve_problem()
#
#     solution_score.append(obj)
#     solve_time.append(time2)
#     current_time = datetime.now()
#     formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
#     print(formatted_time)
#     print(f"instance_{i} have done. \n")
#
#     Gurobi_log_2.info(f"solving {i} instance take {time2}s.")
#     Gurobi_log_2.info(f"{i} instance score: {obj}.")
#
#
# average_time = sum(solve_time) / 36
# average_score = sum(solution_score) / 36
#
# Gurobi_log_2.info(f"----------------------------------------------------------")
# Gurobi_log_2.info(f"average solving time is {average_time}s.")
# Gurobi_log_2.info(f"average solution score is {average_score}.")
# Gurobi_log_2.info(f"score list is {solution_score}s.")
# Gurobi_log_2.info(f"time list is {solve_time}.")
