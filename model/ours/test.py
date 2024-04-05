##########################################################################################
# import
import json
import os
import sys
import logging
from utils import create_logger, copy_all_src
from VLRP_Tester import LRPTester as Tester


##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# parameters
env_params = {
    'depot_num': 5,
    'distribution_point_num': 15,
    'customer_num': 100,
    'instance': 'instance/beta_instance_5_15_100.json',
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
    'depot_num': 5,
    'distribution_point_num': 15,
    'customer_num': 100,
    'phase_size': 2,
}
model_params['depot_num'] = env_params['depot_num']
model_params['distribution_point_num'] = env_params['distribution_point_num']
model_params['customer_num'] = env_params['customer_num']

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': './pre_training_model',  # directory path of pre-trained model and log files saved.
        'epoch': 100,  # epoch version of pre-trained model to load.
    },
    'test_episodes': 1,
    'test_batch_size': 1,
    'augmentation_enable': False,
    'aug_factor': 8,
    'aug_batch_size': 1,
    'test_data_load': {
        'enable': False,
        'filename': '../vrp100_test_seed1234.pt'
    },
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']


logger_params = {
    'log_file': {
        'desc': 'test_cvrp100',
        'filename': 'log.txt'
    }
}


##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    tester = Tester(env_params=env_params, model_params=model_params, tester_params=tester_params)

    copy_all_src(tester.result_folder)

    logger = logging.getLogger('root')

    score, time1, score_list, time_list, node_list = tester.run()

    logger.info(" Selected list: {} ".format(node_list))
    logger.info(" Score list: {} ".format(score_list))
    logger.info(" Time list: {}".format(time_list))
    logger.info(" Average Score: {:.4f} ".format(score))
    logger.info(" Average Time: {:.4f}".format(time1))

    selected_list = []
    for list1 in node_list:
        selected_list.append(list1.tolist())

    # with open('result/15-25-300_node_list.json', "w") as json_file:
    #     json.dump(selected_list, json_file)


def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 1


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":
    main()
