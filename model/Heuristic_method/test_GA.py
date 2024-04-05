import time
from datetime import datetime
import json
import logging
from GA_model import GA_MODEL


def load_instance_from_file(filename):
    with open(filename, 'r') as file:
        instances = json.load(file)
    return instances


ga_params = {
    'depot_num': 2,
    'dp_num': 5,
    'customer_num': 30,

    'light_vehicle_cost': 20,
    'heavy_vehicle_cost': 50,
    'depot_cost': 500,
    'dp_cost': 200,
    'per_distance_cost': 1,

    'population_num': 200,
    'max_generations': 500,
    'tournament_size': 5,
    'crossover_prob': 0.8,
    'mutation_prob': 0.15
}

# define logger
Formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

GA_log_4 = logging.getLogger('GA_for_n300')
GA_log_4.setLevel(logging.INFO)
GA_handler_4 = logging.FileHandler('result/GA_for_table_3.log')
GA_handler_4.setFormatter(Formatter)
GA_log_4.addHandler(GA_handler_4)

def test_GA(file_path):
    print(f"Now test GA for {file_path}")
    GA_log_4.info(f"test GA with {file_path}")

    instances_4 = load_instance_from_file(file_path)
    if len(instances_4[0]) == 64:
        ga_params['depot_num'] = 4
        ga_params['dp_num'] = 10
        ga_params['customer_num'] = 50
    elif len(instances_4[0]) == 120:
        ga_params['depot_num'] = 5
        ga_params['dp_num'] = 15
        ga_params['customer_num'] = 100

    instance_num = len(instances_4)
    solu_score = []
    solve_time = []
    for i in range(instance_num):
        start_time = time.time()
        ga_model = GA_MODEL(**ga_params)
        solu_i, solu_i_score = ga_model.genetic_algorithm(instances_4[i])
        end_time = time.time()
        exec_time = end_time - start_time

        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        print(formatted_time)
        print(f"instance_{i} have done. \n")

        solu_score.append(solu_i_score)
        solve_time.append(exec_time)

        GA_log_4.info(f"solving {i} instance take {exec_time}s.")
        GA_log_4.info(f"{i} instance score: {solu_i_score}.")

    average_time = sum(solve_time) / instance_num
    average_score = sum(solu_score) / instance_num

    GA_log_4.info(f"----------------------------------------------------------")
    GA_log_4.info(f"score list is {solu_score}s.")
    GA_log_4.info(f"time list is {solve_time}s.")
    GA_log_4.info(f"average solving time is {average_time}s.")
    GA_log_4.info(f"average solution score is {average_score}.\n")

