
import torch
import time
import os
from logging import getLogger

from LRP_2E_Env import LRP_Env as Env
from LRP_2E_Model import LRP_Model as Model

from utils import *


def load_instance_from_file(filename):
    with open(filename, 'r') as file:
        instances = json.load(file)
    return instances

class LRPTester:
    def __init__(self, env_params, model_params, tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.instance = env_params['instance']

        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()


    def run(self):
        instance = load_instance_from_file(self.instance)

        score_list = []
        time_list = []
        node_list = []

        for i in range(len(instance)):
            start_time = time.time()
            batch_size = self.tester_params['test_batch_size']
            score, selected_node = self._test_one_batch(batch_size, instance[i])
            time1 = time.time() - start_time
            score_list.append(score)
            time_list.append(time1)
            node_list.append(selected_node)

        average_score = sum(score_list) / len(score_list)
        average_time = sum(time_list) / len(time_list)

        return average_score, average_time, score_list, time_list, node_list

    def _test_one_batch(self, batch_size, instance):

        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            # self.env.load_problems(batch_size)
            self.env.load_instance(instance)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            selected, _ = self.model(state)
            # shape: (batch, pomo)
            state, reward, done, selected_node = self.env.step(selected)

        # Return
        ###############################################
        aug_reward = reward.reshape(aug_factor, batch_size, self.env.lrp_size)
        # shape: (augmentation, batch, pomo)

        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value

        return no_aug_score.item(), selected_node
