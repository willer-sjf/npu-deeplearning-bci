import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Q_Net(nn.Module):
    
    def __init__(self,v_dim):
        super(Q_Net,self).__init__()
        
        self.v_dim = v_dim
        self.pre_feature = nn.Sequential(
            nn.Linear(v_dim, 64),
            nn.Tanh(),
        )
        self.first_order_feature= nn.Sequential(
            nn.Linear(v_dim, 64),
            nn.Tanh(),
        )
        self.second_order_feature = nn.Sequential(
            nn.Linear(v_dim, 64),
            nn.Tanh(),
        )
        self.fc1 = nn.Linear(64*3,64)
        self.fc2 = nn.Linear(64  , 1)
    def forward(self, feature, mean_feature, var_feature):
        
        f_pre = self.pre_feature(feature)
        f_fst = self.first_order_feature(mean_feature - feature)
        f_scd = self.second_order_feature(var_feature)
        
        f_merge = torch.cat([f_pre, f_fst, f_scd], 1)
        q = F.relu(self.fc1(f_merge))
        q = self.fc2(q)
        return q
        
