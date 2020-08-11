import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import datetime
import random 


def cal_cosine_similarity(vector_a, vector_b):
    inner = np.dot(vector_a, vector_b.transpose())
    norm = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    return inner / norm

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
        self.index = new_index
        return avg_state, ret_state
    
    def mean_var_state(self, remaining_feature):
        shape = remaining_feature.shape[1]
        size  = remaining_feature.shape[0]
        mean_state = torch.zeros(shape, dtype=torch.float32)
        var_state  = torch.zeros(shape, dtype=torch.float32)
        for i in range(size):
            mean_state += remaining_feature[i]
        mean_state /= size
        for i in range(size):
            var_state += torch.pow((remaining_feature[i] - mean_state), 2)
        var_state /= size
        return mean_state, var_state
    
    
class DropEnv:
    def __init__(self, data, label):
        self.reward_module = RewardNet(input_size=data.shape[-1])
        self.data  = torch.from_numpy(data)
        self.label = label
        
        self.phase = 'test'
        self.gamma = 0.1
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
        if reward < 0 and random.random() < 0.5:
            done = True
        return state, reward, done
    
    def train(self):
        self.phase = 'train'
        
    def eval(self):
        self.phase = 'test'
        
    def reset(self):
        index = random.randint(1, self.data_size-1)
        single_data = self.data[index]
        self.manager = FeatureManager(single_data)
        self.current_label = self.label[index]
        
        if self.phase == 'train':
            random_drop_num = random.randint(1, self.channel_size//3)
            random_drop_idx = random.sample(range(0, self.channel_size), random_drop_num)
            for idx in random_drop_idx:
                self.manager.drop(idx)
                
        init_avg_state, init_state = self.manager.state()
        #init_avg_state = torch.FloatTensor(init_avg_state)
        
        init_cls_vector = self.reward_module(init_avg_state).numpy()
        self.old_sim = cal_cosine_similarity(init_cls_vector, self.current_label)
        return init_state
