##########################################################################################
# import
import os
import sys
import logging
from utils import create_logger, copy_all_src
from Tester import LRPTester as Tester


##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 3


##########################################################################################
# Path Config

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# parameters
env_params = {
    'depot_num': 1,
    'distribution_point_num': 5,
    'customer_num': 100,
    'instance': 'instance/coord100-5-2-2e.dat',
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
    'depot_num': 1,
    'distribution_point_num': 10,
    'customer_num': 100,
    'phase_size': 2,
}
model_params['distribution_point_num'] = env_params['distribution_point_num']

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

sa_params = {
    'depot_num': 1,
    'dp_num': 10,
    'customer_num': 100,
    'node_num': 111,

    'light_vehicle_cost': 1000,
    'heavy_vehicle_cost': 5000,
    'dp_cost': 200,
    'per_distance_cost': 100,

    'init_temperature': 1000,
    'cooling_rate': 0.999,
    'max_iteration': 50000

}
sa_params['dp_num'] = model_params['distribution_point_num']
sa_params['node_num'] = sa_params['depot_num'] + sa_params['dp_num'] + sa_params['customer_num']

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

    tester = Tester(env_params=env_params, model_params=model_params, tester_params=tester_params, sa_params=sa_params)

    copy_all_src(tester.result_folder)

    logger = logging.getLogger('root')

    score, time1 = tester.run()

    logger.info(" Average Score: {:.4f} ".format(score))
    logger.info(" Average Time: {:.4f}".format(time1))


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
