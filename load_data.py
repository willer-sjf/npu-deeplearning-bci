import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from scipy.fftpack import fft, ifft
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import MinMaxScaler,StandardScaler

FRE_THETA = 4
FRE_ALPHA = 8
FRE_BETA  = 13
FRE_CUT   = 30
FRE_SAMPLE= 336
ORDER     = 8
PATH = "C:\\Users\\Willer\\Desktop\\Development\\JupyterLab\\word_EEG\\"

np.set_printoptions(threshold=np.inf)

def getTest(channel:int, pca=True, std=True):
    dic = {}
    for num in range(12, 13):
        df = pd.read_csv(PATH + "splitted_word_with_label{}.csv" .format(num))
        array = df.values
        i = 0
        while i < len(array):
            data = np.array(array[i:i+315,3:11])   
            if pca:         
                data = preprocess(data, channel, pca, std)

            data_sing = []
            data_line = []
            for col in range(channel):
                data_line = data[:, col].reshape(-1)
                data_sing.append(data_line)
            dic[array[i][-1]] = (data_sing,array[i][-2])
            i += 315
    return dic

def loadFiltData(channel:int, pca=True, std=True, size=12):
    data = []
    labels = []
    for i in range(1, size):
        
        df_label = pd.read_csv(PATH + "splitted_word_with_label{}.csv" .format(i))
        df_data  = pd.read_csv(PATH + "lowpass_word{}.csv" .format(i), header=None)
        array_label = df_label.values
        array_data  = df_data.values
        line = 0
        
        while line < len(array_label)-2:
            labels.append([1] if array_label[line+3][-2] == 'confused' else [0])
            #labels.append(1 if array_label[line+3][-2] == 'confused' else 0)
            data_sing = []

            if pca or std:
                data_mat = preprocess(array_data[line:line+315, :8], channel, pca, std)
            else:
                data_mat = array_data[line:line+315, :8]
            
            for col in range(channel):
                data_line = data_mat[:, col].reshape(-1)
                data_sing.append(data_line)
            # np_line = np.array(data_sing)
            # np_line = np_line.reshape(-1)
            # data.append(np_line)
            data.append(data_sing)
            line += 315
            
    return train_test_split(data,labels,test_size=0.2)

def loadData(channel:int, pca=True, std=True, size=12):
    data = []
    labels = []
    for i in range(1, size):
        
        df = pd.read_csv(PATH + "splitted_word_with_label{}.csv" .format(i))
        array = df.values

        line = 0

        while line < len(array):
            labels.append([1] if array[line][-2] == 'confused' else [0])
            data_sing = []

            if pca or std:
                data_mat = preprocess(array[line:line+315, 3:11], channel, pca, std)
            else:
                data_mat = array[line:line+315, 3:11]
            
            for col in range(channel):
                data_line = data_mat[:, col].reshape(-1)
                data_sing.append(data_line)
            data_sing = filter(data_sing)
            data.append(data_sing)
            line += 315
            
    return train_test_split(data,labels,test_size=0.2)

def filter(data):
    cb, ca = signal.butter(ORDER,  2 * FRE_CUT/FRE_SAMPLE  , 'lowpass'   ) 
    lb, la = signal.butter(ORDER,  2 * FRE_THETA/FRE_SAMPLE, 'highpass'  ) 
    mub, mua = signal.butter(ORDER,  2 * FRE_ALPHA/FRE_SAMPLE, 'highpass') 
    mdb, mda = signal.butter(ORDER,  2 * FRE_ALPHA/FRE_SAMPLE, 'lowpass' )
    hub, hua = signal.butter(ORDER,  2 * FRE_BETA/FRE_SAMPLE , 'highpass') 
    hdb, hda = signal.butter(ORDER,  2 * FRE_BETA/FRE_SAMPLE , 'lowpass' ) 

    data = signal.filtfilt(cb, ca, data)
    data = signal.filtfilt(lb, la, data)

    theta = preprocess_conv(signal.filtfilt(mdb, mda, data))
    alpha = preprocess_conv(signal.filtfilt(hdb,hda,signal.filtfilt(mub, mua, data))) * 8
    beta  = preprocess_conv(signal.filtfilt(hub, hua, data)) * 15

    new_data = np.array([theta, alpha, beta])
    return new_data

def preprocess_conv(new_data):
    new_data = (new_data - np.min(new_data)) / (np.max(new_data) - np.min(new_data)) * 255.0
    new_data = np.uint8(new_data)
    return new_data

def preprocess(data, channel, pca, std):

    if std:
        # std = StandardScaler()
        # data = std.fit_transform(data)
        data = data - np.mean(data)
    if pca:
        pcaS = PCA(n_components=channel)
        data = pcaS.fit_transform(data)
    return data

class MyDataset(Dataset):
    def __init__(self, x, y, device='cpu'):
        self.x = x
        self.y = y
        self.len = len(y)
        self.device = device

    def __len__(self):
        return self.len
    
    def __getitem__(self, item):
        npdata = np.ascontiguousarray(self.x[item], dtype=np.float32)
        data = torch.FloatTensor(npdata).to(self.device)
        label = torch.LongTensor(self.y[item]).to(self.device)
        
        return data, label
    
if __name__ == '__main__':
    x,y = loadData(1,pca=True)
    a = np.array(x[0])
    for i in range(10):
        print(np.array(x[i*100]))
        print(y[i])
    print(y[0])

    
        
    

