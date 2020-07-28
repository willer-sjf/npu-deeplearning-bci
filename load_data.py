import os
import numpy as np
import pandas as pd
import pickle
import gumpy
from datetime import datetime

import keras
import keras.utils as ku
import kapre
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
import utils
utils.set_random_seed(0)

PATH = '/home/willer/Desktop/Development/Python/dataset/grazdata/'
def label_map(x, pos=0, classes=2):
    if x[pos] >= 5:
        return [1, 0]
    else:
        return [0, 1]

def get_dataloader_graz(batch_size=128):
    ndata, nlabel = get_grazdata()
    nlabel = nlabel.reshape(-1, 1)
    train_loader, test_loader = boost_dataloader(ndata, nlabel, batch_size=batch_size)
    return train_loader, test_loader
    
def get_dataloader_deap(file_path, batch_size=64, test_size=0.2, task='classify'):
    
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

def get_grazdata():
    ndata, nlabel = get_Xy('B01')
    for i in range(2, 10):
        x, y = get_Xy('B0'+str(i))
        ndata = np.concatenate([ndata, x], 0)
        nlabel = np.concatenate([nlabel, y], 0)
    return ndata, nlabel

def boost_dataloader(data, label, batch_size=128):
    
    train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.20)
    train_set = MyDataset(train_data, train_label)
    test_set  = MyDataset(test_data, test_label)
    
    train_loader = DataLoader(train_set, batch_size=batch_size)
    test_loader  = DataLoader(test_set , batch_size=batch_size)
    
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

        data = torch.FloatTensor(data).to(self.device)
        if self.task == 'classify':
            label = torch.LongTensor(label).to(self.device)
        else:
            label = torch.FloatTensor(label).to(self.device)
        return data, label
    
def load_preprocess_data(data, debug, lowcut, highcut, w0, Q, anti_drift, class_count, cutoff, axis, fs):
    """Load and preprocess data.

    The routine loads data with the use of gumpy's Dataset objects, and
    subsequently applies some post-processing filters to improve the data.
    """
    # TODO: improve documentation

    data_loaded = data.load()

    if debug:
#         print('Band-pass filtering the data in frequency range from %.1f Hz to %.1f Hz... '
#           %(lowcut, highcut))

        data_notch_filtered = gumpy.signal.notch(data_loaded.raw_data, cutoff, axis)
        data_hp_filtered = gumpy.signal.butter_highpass(data_notch_filtered, anti_drift, axis)
        data_bp_filtered = gumpy.signal.butter_bandpass(data_hp_filtered, lowcut, highcut, axis)

        # Split data into classes.
        # TODO: as soon as gumpy.utils.extract_trails2 is merged with the
        #       regular extract_trails, change here accordingly!
        class1_mat, class2_mat = gumpy.utils.extract_trials2(data_bp_filtered, data_loaded.trials,
                                                             data_loaded.labels, data_loaded.trial_total,
                                                             fs, nbClasses = 2)

        # concatenate data for training and create labels
        x_train = np.concatenate((class1_mat, class2_mat))
        labels_c1 = np.zeros((class1_mat.shape[0], ))
        labels_c2 = np.ones((class2_mat.shape[0], ))
        y_train = np.concatenate((labels_c1, labels_c2))

        # for categorical crossentropy
#         y_train = ku.to_categorical(y_train)

#         print("Data loaded and processed successfully!")
        return x_train, y_train
    
def get_Xy(sub="B01"):
    DEBUG = True
    CLASS_COUNT = 2
    DROPOUT = 0.2   # dropout rate in float

    # parameters for filtering data
    FS = 250
    LOWCUT = 2
    HIGHCUT = 60
    ANTI_DRIFT = 0.5
    CUTOFF = 50.0 # freq to be removed from signal (Hz) for notch filter
    Q = 30.0  # quality factor for notch filter 
    W0 = CUTOFF/(FS/2)
    AXIS = 0

    #set random seed
    SEED = 42
    KFOLD = 5
    # ## Load raw data 
    # Before training and testing a model, we need some data. The following code shows how to load a dataset using ``gumpy``.
    # specify the location of the GrazB datasets
    data_dir = PATH

    subject = sub

    # initialize the data-structure, but do _not_ load the data yet
    grazb_data = gumpy.data.GrazB(data_dir, subject)

    # now that the dataset is setup, we can load the data. This will be handled from within the utils function, 
    # which will first load the data and subsequently filter it using a notch and a sbandpass filter.
    # the utility function will then return the training data. 取得每一次试验的所有数据，8s。
    x_train, y_train = load_preprocess_data(grazb_data, True, LOWCUT, HIGHCUT, W0, Q, ANTI_DRIFT, CLASS_COUNT, CUTOFF, AXIS, FS)


    # ## Augment data

    x_augmented, y_augmented = gumpy.signal.sliding_window(data = x_train[:,:,:],
                                                              labels = y_train[:],
                                                              window_sz = 4 * FS,
                                                              n_hop = FS // 5,
                                                              n_start = FS * 1)
    x_subject = x_augmented
    y_subject = y_augmented
    x_subject = np.rollaxis(x_subject, 2, 1)
    
    
    return x_subject,y_subject


def print_version_info():
    now = datetime.now()
    print('%s/%s/%s' % (now.year, now.month, now.day))
    print('Keras version: {}'.format(keras.__version__))
    if keras.backend.backend() == 'tensorflow':
        import tensorflow
        print('Keras backend: {}: {}'.format(keras.backend.backend(), tensorflow.__version__))
    else:
        import theano
        print('Keras backend: {}: {}'.format(keras.backend.backend(), theano.__version__))
    print('Keras image dim ordering: {}'.format(keras.backend.image_data_format()))
    print('Kapre version: {}'.format(kapre.__version__))


def get_Xy2(sub="B01"): # _v2
    eeg_data = {}
    dataset_dir = PATH
    
    file = os.path.join(dataset_dir, f"sub{sub}-win4-stride0.2 V2.pkl")

    if not os.path.exists(file):
        x_subject,y_subject=get_Xy(sub)
        eeg_data["x_subject"]=x_subject
        eeg_data["y_subject"]=y_subject
        with open(file, 'wb') as fw:
            pickle.dump(eeg_data, fw, protocol=4)
    else:
        with open(file, 'rb') as fr:
            eeg_data = pickle.load(fr)

    return eeg_data["x_subject"],eeg_data["y_subject"]
    
def get_Xy_v1(sub="B01"):
    eeg_data = {}
    dataset_dir = PATH
    
    file = os.path.join(dataset_dir, f"sub{sub}-win4-stride0.2 V2.pkl")

    if not os.path.exists(file):
        x_subject,y_subject=get_Xy(sub)
        eeg_data["x_subject"]=x_subject
        eeg_data["y_subject"]=y_subject
        
        with open(file, 'wb') as fw:
            pickle.dump(eeg_data, fw, protocol=4)
    else:
        with open(file, 'rb') as fr:
            eeg_data = pickle.load(fr)
    return eeg_data["x_subject"], ku.to_categorical(eeg_data["y_subject"])