import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import utils

utils.set_random_seed(0)

class MyDataset(Dataset):
    def __init__(self, data_file, label_file):
        self.data  = pd.read_csv(data_file)
        self.label = pd.read_csv(label_file)
        self.device = torch.device('cuda' if torch.cuda.is_availiable() else 'cpu')
        
    def __len__(self):
        return self.len
        
    def __getitem__(self, index):
        data  = self.data.iloc[index*40:(index+1)*40]
        label = self.label.iloc[index]
        
        data  = torch.FloatTensor(data).to(self.device)
        label = torch.FloatTensor(data).to(self.device)
        return data, label