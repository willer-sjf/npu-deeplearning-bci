import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FRE_THETA  = 4
FRE_ALPHA  = 8
FRE_BETA   = 13
FRE_CUT    = 30
FRE_SAMPLE = 512
ORDER      = 8

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
def freq_filter(data):
    cb, ca = signal.butter(ORDER,  2 * FRE_CUT/FRE_SAMPLE  , 'lowpass'   ) 
    lb, la = signal.butter(ORDER,  2 * FRE_THETA/FRE_SAMPLE, 'highpass'  ) 
    mub, mua = signal.butter(ORDER,  2 * FRE_ALPHA/FRE_SAMPLE, 'highpass') 
    mdb, mda = signal.butter(ORDER,  2 * FRE_ALPHA/FRE_SAMPLE, 'lowpass' )
    hub, hua = signal.butter(ORDER,  2 * FRE_BETA/FRE_SAMPLE , 'highpass') 
    hdb, hda = signal.butter(ORDER,  2 * FRE_BETA/FRE_SAMPLE , 'lowpass' ) 

    data = signal.filtfilt(cb, ca, data)
    data = signal.filtfilt(lb, la, data)

    theta = preprocess(signal.filtfilt(mdb, mda, data))
    alpha = preprocess(signal.filtfilt(hdb,hda,signal.filtfilt(mub, mua, data))) 
    beta  = preprocess(signal.filtfilt(hub, hua, data)) 

    new_data = np.array([theta, alpha, beta])
    return new_data