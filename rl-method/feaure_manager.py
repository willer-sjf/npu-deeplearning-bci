import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import datetime
import random 


class FeatureManager:
    def __init__(self, data):
        self.data = data
        self.size = data.shape[1]
        self.index = list(range(data.shape[0]))

    def drop(self, action):
        self.index.remove(action)
    
    def state(self):
        new_state = self.data[self.index]
        new_size  = new_state.shape[0]
        new_index = list(range(new_size))
        
        avg_state = torch.sum(new_state, 0) / new_size
        ret_state = []
        for i in range(new_size):
            remaining = new_state[new_index[:i] + new_index[i+1:]]
            mean_state, var_state = self.mean_var_state(remaining)
            ret_state.append([new_state[i], mean_state, var_state])
        
        return avg_state, ret_state
    
    def mean_var_state(self, remaining_feature):
        shape = remaining_feature.shape[1]
        size  = remaining_feature.shape[0]
        mean_state = torch.zeros(shape, dtype=float)
        var_state  = torch.zeros(shape, dtype=float)
        for i in range(size):
            mean_state += remaining_feature[i]
        mean_state /= size
        for i in range(size):
            var_state += torch.pow((remaining_feature[i] - mean_state), 2)
        var_state /= size
        return mean_state, var_state
    
    
class DropEnv:
    def __init__(self, data, label):
        self.reward_module = RewardNet(input_size=6)
        self.data  = data
        self.label = label
        
        self.data_size = self.data.shape[0]
        self.channel_size = self.data.shape[1]
        
    def step(self, action):
        self.manager.drop(action)
        avg_state, state = self.manager.state()

        reward = self.reward_module(avg_state).data
        reward = reward[0] - reward[1]

        done = False
        if reward < 0 and random.random() < 0.5:
            done = True
        return state, reward, done
    
    def reset(self):
        single_data = data[random.randint(1, self.data_size-1)]
        self.manager = FeatureManager(single_data)
#         random_drop_num = random.randint(1, self.channel_size//3)
#         random_drop_idx = random.sample(range(0, self.channel_size), random_drop_num)
#         for idx in random_drop_idx:
#             self.manager.drop(idx)
        return self.manager.state()