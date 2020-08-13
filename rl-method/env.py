import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import datetime
import random
import preprocess
import gc

def cal_cosine_similarity(vector_a, vector_b):
    inner = np.dot(vector_a, vector_b.transpose())
    norm = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    return inner / norm

class FeatureManager:
    def __init__(self, data):
        self.data = data
        self.size = data.shape[-1]
        self.index = list(range(data.shape[0]))

    def drop(self, action):
        self.index.remove(self.index[action])

    def state(self):
        new_state = self.data[self.index]
        new_size  = new_state.shape[0]
        new_index = list(range(new_size))

        avg_state = torch.mean(new_state, 0)
        ret_state = []
        for i in range(new_size):
            remaining = new_state[new_index[:i] + new_index[i+1:]]
            mean_state, var_state = self.mean_var_state(remaining)
            ret_state.append([new_state[i], mean_state, var_state])
        return avg_state, ret_state

    def mean_var_state(self, remaining_feature):
        shape = remaining_feature.shape[1]
        size  = remaining_feature.shape[0]
        mean_state = torch.mean(remaining_feature, 0)
        var_state  = torch.mean(torch.pow(remaining_feature - mean_state, 2), 0)
        return mean_state, var_state


class DropEnv:
    def __init__(self, data, label, reward_model_path):

        self.gamma = 0.1
        self.random_stop = 0.5

        self.data  = torch.from_numpy(data)
        self.label = label
        self.reward_module = preprocess.get_reward_net(data[0].shape[-1], reward_model_path)

        self.random_drop = True
        self.data_size = self.data.shape[0]
        self.channel_size = self.data.shape[1]

    def step(self, action):
        self.manager.drop(action)
        avg_state, state = self.manager.state()
        cls_vector = self.reward_module(avg_state).numpy()
        self.new_sim = cal_cosine_similarity(cls_vector, self.current_label)
        reward  = self.new_sim - self.old_sim + self.gamma
        self.old_sim = self.new_sim

        done = False
        if reward < 0 and (self.random_drop == True and random.random() < self.random_stop or self.random_drop == False):
                done = True
        return state, reward, done

    def train(self):
        self.random_drop = True

    def eval(self):
        self.random_drop = False

    def reset(self):
        index = random.randint(0, self.data_size-1)
        single_data = self.data[index]
        self.current_label = self.label[index]
        self.manager = FeatureManager(single_data)

        if self.random_drop:
            random_drop_num = random.randint(1, self.channel_size//4)
            random_drop_idx = random.sample(range(0, self.channel_size - random_drop_num - 1), random_drop_num)
            for idx in random_drop_idx:
                self.manager.drop(idx)

        init_avg_state, init_state = self.manager.state()
        init_cls_vector = self.reward_module(init_avg_state).numpy()
        self.old_sim = cal_cosine_similarity(init_cls_vector, self.current_label)
        gc.collect()
        return init_state
