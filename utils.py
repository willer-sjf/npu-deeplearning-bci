import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyeeg
import pywt

def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    x = exp_x / np.sum(exp_x)
    return x

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def freq_filter(data):
    FRE_THETA  = 4
    FRE_ALPHA  = 8
    FRE_BETA   = 13
    FRE_CUT    = 30
    FRE_SAMPLE = 512
    ORDER      = 8

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

def wavelet_transform(data):
    coeffs = pywt.wavedec(data, 'db4',level=4)
    cA,cB,cC,cD,cE = coeffs
    data = np.concatenate((cA, cB), axis=0)

    data = normalize(data, axis=1, copy=False)

    f, t, z = signal.stft(data, fs=128)
    z = np.abs(z)

    h = np.sum(z[:, 6:14 , :], axis=1)
    b = np.sum(z[:, 14:26, :], axis=1)
    p = np.sum(z[:, 26:58, :], axis=1)
    t = np.sum(z[:, 58:94, :], axis=1)
    data = np.concatenate((h, b, p, t), 0)
    return data

def freq_filter_shorttime(data, channel=3, sequence=1000):
    """
    pyeeg bin power band filter
    """
    band = [4,8,12,16,25,45]
    window_size = 256
    step_size = 64
    sample_rate = 128
    size = data.shape[0]
    new_data = np.zeros(shape=(40*32, 160, 123))
    for index in range(size):
        data = ndata[index][:32]
        fp_data = np.zeros(shape=(123, 160), dtype=float)
        for t in range(123):
            fp_data_t = np.zeros(160, dtype=float)
            for i in range(32):
                single = data[i][t:t + window_size]
                fp, norm_fp = pyeeg.bin_power(single, band, sample_rate)
                fp_data_t[i*5:(i+1)*5] = norm_fp
            fp_data[t] = fp_data_t
        fp_data = np.swapaxes(fp_data, 0, 1)
        new_data[index] = fp_data
    return new_data

def integrate_data(PATH):
    """
    Low Effectiveness
    """
    total_data  = None
    total_label = None
    for i in range(1,33):
        if i < 10:
            expId = "0" + str(i)
        else:
            expId = str(i)
        file_path = PATH + 's' + str(expId) + '.dat'
        tmp_data  = pickle.load(open(file_path, 'rb'), encoding='bytes')
        data  = tmp_data[b'data']
        label = tmp_data[b'labels']

        #data = np.swapaxes(data, 1, 2)
        data = data.reshape(40*40, 8064)

        if str(type(total_data)) == "<class 'numpy.ndarray'>":
            total_data  = np.concatenate((total_data, data), axis=0)
            total_label = np.concatenate((total_label, label), axis=0)
        else:
            total_data  = data
            total_label = label

    pdata  = pd.DataFrame(total_data)
    plabel = pd.DataFrame(total_label)
    pdata.to_csv( PATH + 'preprocessed_data_1d.csv', index=False)
    plabel.to_csv(PATH + 'preprocessed_label_1d.csv', index=False)
