import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import utils

utils.set_random_seed(0)

def label_map(x, pos=0, classes=2):
    if x[pos] >= 5:
        return [1, 0]
    else:
        return [0, 1]
    
def get_dataloader(file_path, batch_size=64, test_size=0.2, task='classify'):
    
    pdata  = pd.read_csv(file_path + 'preprocessed_data_1d.csv')
    plabel = pd.read_csv(file_path + 'preprocessed_label_1d.csv')
    ndata = np.array(pdata)
    ndata = ndata.reshape(40*32, 40, 8064)
    if task == 'classify':
        nlabel = np.array(plabel.apply(label_map, axis=1))
    else:
        nlabel = np.array(plabel)
    train_data, test_data, train_label, test_label = train_test_split(ndata, nlabel, test_size=0.20)
    train_set = MyDataset(train_data, train_label)
    test_set  = MyDataset(test_data, test_label)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_set , batch_size=batch_size, shuffle=True)
    
    return train_loader, test_loader


class MyDataset(Dataset):
    def __init__(self, data, label, task='classify'):
        self.data  = data
        self.label = label
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.task = task
        if task not in ['classify', 'regression']:
            raise ValueError("UNDEFINED TASK")
            
    def __len__(self):
        return self.label.shape[0]
        
    def __getitem__(self, index):
        data  = self.data[index]
        label = self.label[index]
        
        data  = torch.FloatTensor(data).to(self.device)
        if self.task == 'classify':
            label = torch.LongTensor(label).to(self.device)
        else:
            label = torch.FloatTensor(label).to(self.device)
        return data, label