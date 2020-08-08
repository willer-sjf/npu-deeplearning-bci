import os
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def useless():
    data = pyedflib.EdfReader(PATH + "S001R01.edf")
    n = data.signals_in_file
    print("signal numbers:", n)

    raw_data = mne.io.read_raw_edf(PATH, preload=True)   
    pdata = raw_data.to_data_frame()
    raw.pick_channels([signal_name])   
    eeg = raw.to_data_frame()   # 将读取的数据转换成pandas的DataFrame数据格式
    eeg = list(eeg.values[:,1])  

    events_from_annot, event_dict = mne.events_from_annotations(raw_data)
    print(event_dict)
    print(events_from_annot)

    anno = data.read_annotation()
    
def boost_dataloader(data, label, batch_size=128):

    train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.20)
    train_set = MyDataset(train_data, train_label)
    test_set  = MyDataset(test_data, test_label)

    train_loader = DataLoader(train_set, batch_size=batch_size)
    test_loader  = DataLoader(test_set , batch_size=batch_size)

    return train_loader, test_loader