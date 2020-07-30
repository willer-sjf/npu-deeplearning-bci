import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import utils
import load_data
import datetime
import model
from sklearn.preprocessing import normalize

seed = 0
utils.set_random_seed(seed)
np.set_printoptions(precision=4, suppress=True)
device = torch.device('cuda')
PATH = '/home/willer/Desktop/Development/Python/MyRepo/npu-deeplearning-bci/model/'


ndata, nlabel = load_data.get_grazdata()
train_loader, test_loader = load_data.get_dataloader_graz()
fnet = model.FrequencyAttentionNet().to(device)
fnet.load_state_dict(torch.load(PATH + 'FrequencyAttentionNet_feature_withLSTM.pkl'))
snet = model.SpacialAttentionNet().to(device)
snet.load_state_dict(torch.load(PATH + 'SpacialAttentionNet_feature_withLSTM.pkl'))
tnet = model.TemporalAttentionNet().to(device)
tnet.load_state_dict(torch.load(PATH + 'TemporalAttentionNet_feature_withLSTM.pkl'))


def get_attention_distribution(f_size, t_size, s_size):
    f_1 = np.zeros(3, dtype=np.float)
    f_0 = np.zeros(3, dtype=np.float)
    t_1 = np.zeros(10, dtype=np.float)
    t_0 = np.zeros(10, dtype=np.float)
    s_1 = np.zeros(3, dtype=np.float)
    s_0 = np.zeros(3, dtype=np.float)

    l1 = 0
    l0 = 0
    with torch.no_grad():
        for input, label in test_loader:

            foutput, fattn = fnet(input)
            soutput, sattn = snet(input)
            toutput, tattn = tnet(input)

            label = label.cpu().numpy()
            fattn = fattn.cpu().numpy()
            sattn = sattn.cpu().numpy()
            tattn = tattn.cpu().numpy()

            index_1 = np.where(label==1)[0]
            index_0 = np.where(label==0)[0]

            f_1 += fattn[index_1].sum(0)
            f_0 += fattn[index_0].sum(0)
            s_1 += sattn[index_1].sum(0)
            s_0 += sattn[index_0].sum(0)
            t_1 += tattn[index_1].sum(0)
            t_0 += tattn[index_0].sum(0)
            l1 += index_1.shape[0]
            l0 += index_0.shape[0]
            
index = -31235
with torch.no_grad():
    input = torch.FloatTensor([ndata[index]]).to(device)
    o, attn = fnet(input)
    print(attn)