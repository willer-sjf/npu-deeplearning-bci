import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
sys.path.append("..")
from model import ResidualBlock1d, SubConvNet

def encode_data(model_path):

    enet = EncodeNet()
    pnet = PretrainNet()
    pnet.load_state_dict(torch.load(model_path))

    enet_dict = enet.state_dict()
    for (name, param) in enet_dict.items():
        enet_dict[name] = copy.deepcopy(pnet.state_dict()[name])
    enet.load_state_dict(enet_dict)
    enet.eval()
    enet.to(torch.device('cuda'))
    # ===================
    # Get your Data
    # ===================
    rdata  = None
    rlabel = None
    n_classes = 2

    train_loader, test_loader = utils.boost_dataloader(rdata, rlabel, batch_size=512)
    with torch.no_grad():
        for input, label in train_loader:
            output = enet(input).cpu().numpy()
            label = label.cpu().numpy().reshape(-1)
            vec_label = np.eye(n_classes)[label]
            if str(type(ndata)) == "<class 'NoneType'>":
                ndata  = output
                nlabel = vec_label
            else:
                ndata  = np.concatenate([ndata, output], 0)
                nlabel = np.concatenate([nlabel, vec_label], 0)

        for input, label in test_loader:
            output = enet(input).cpu().numpy()
            label = label.cpu().numpy().reshape(-1)
            vec_label = np.eye(n_classes)[label]

            ndata  = np.concatenate([ndata, output], 0)
            nlabel = np.concatenate([nlabel, vec_label], 0)

    return ndata, nlabel

def train_pretrainnet(save_path):
    pass

def get_reward_net(input_size, model_path):

    rnet = RewardNet(input_size)
    pnet = PretrainNet_T(hidden_size=256)
    pnet.load_state_dict(torch.load(model_path))

    rnet_dict = rnet.state_dict()
    for (name, param)  in rnet_dict.items():
        rnet_dict[name] = copy.deepcopy(pnet.state_dict()[name])
    rnet.load_state_dict(rnet_dict)
    rnet.eval()
    return rnet


class PretrainNet(nn.Module):
    def __init__(
        self,
        in_channel=3,
        sequence_lens=1000,
        time_lens=10,
        hidden_size=64,
        output_size=2,
        layer_size=1,
        bidirectional=True
    ):
        super(PretrainNet, self).__init__()

        if sequence_lens % time_lens != 0:
            raise ValueError("Invalid time lens")

        self.in_channel  = in_channel
        self.time_lens   = time_lens
        self.hidden_size = hidden_size
        self.layer_size  = layer_size
        self.window_size = sequence_lens // time_lens
        self.device      = torch.device('cuda')

        self.subconv = SubConvNet(in_channel=1, out_channel=2)
        self.input_size = self._adaptive_feature_size()

        self.lstm = nn.LSTM(self.input_size, hidden_size, layer_size, bidirectional=bidirectional)
        if bidirectional:
            self.layer_size *= 2

        self.fn1 = nn.Linear(hidden_size * self.layer_size, 128)
        self.fn2 = nn.Linear(128, output_size)

    def forward(self, x):

        batch_size = x.shape[0]

        x = x.chunk(self.time_lens, 2)
        x = torch.stack(x, 2)
        x = x.reshape(batch_size * self.in_channel * self.time_lens, 1, self.window_size)

        x = self.subconv(x)
        x = x.view(batch_size * self.in_channel, self.time_lens, self.input_size)
        x = x.permute(1, 0, 2)


        h_0 = torch.zeros(self.layer_size, batch_size * self.in_channel, self.hidden_size).to(self.device)
        c_0 = torch.zeros(self.layer_size, batch_size * self.in_channel, self.hidden_size).to(self.device)
        x, (h_final, c_final) = self.lstm(x, (h_0, c_0))
        x = x[-1, :, :]

        x = x.view(batch_size, self.in_channel, -1)

        x = x.permute(0, 2, 1)
        x = F.avg_pool1d(x, self.in_channel)
        x = x.view(batch_size, -1)

        x = F.relu(self.fn1(x))
        x = F.softmax(self.fn2(x), dim=-1)
        return x, tx

    def _adaptive_feature_size(self):
        x = torch.zeros(1, 1, self.window_size)
        return self.subconv(x).view(-1).shape[0]

class RewardNet(nn.Module):

    def __init__(self, input_size=128, output_size=2):
        super(RewardNet, self).__init__()
        self.fn1 = nn.Linear(input_size, 128)
        self.fn2 = nn.Linear(128, output_size)

    def forward(self, x):
        with torch.no_grad():
            x = F.relu(self.fn1(x))
            x = F.softmax(self.fn2(x), dim=-1)
            return x

class EncodeNet(nn.Module):

    def __init__(
        self,
        in_channel=3,
        sequence_lens=1000,
        time_lens=10,
        hidden_size=64,
        output_size=2,
        layer_size=1,
        bidirectional=True
    ):
        super(EncodeNet, self).__init__()
        self.in_channel  = in_channel
        self.time_lens   = time_lens
        self.hidden_size = hidden_size
        self.layer_size  = layer_size
        self.window_size = sequence_lens // time_lens
        self.device      = torch.device('cuda')

        self.subconv = SubConvNet(in_channel=1, out_channel=2)
        self.input_size = self._adaptive_feature_size()

        self.lstm = nn.LSTM(self.input_size, hidden_size, layer_size, bidirectional=bidirectional)
        if bidirectional:
            self.layer_size *= 2

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.chunk(self.time_lens, 2)
        x = torch.stack(x, 2)
        x = x.reshape(batch_size * self.in_channel * self.time_lens, 1, self.window_size)

        x = self.subconv(x)
        x = x.view(batch_size * self.in_channel, self.time_lens, self.input_size)
        x = x.permute(1, 0, 2)

        h_0 = torch.zeros(self.layer_size, batch_size * self.in_channel, self.hidden_size).to(self.device)
        c_0 = torch.zeros(self.layer_size, batch_size * self.in_channel, self.hidden_size).to(self.device)
        x, (h_final, c_final) = self.lstm(x, (h_0, c_0))
        x = x[-1, :, :]

        x = x.view(batch_size, self.in_channel, -1)
        return x

    def _adaptive_feature_size(self):
        x = torch.zeros(1, 1, self.window_size)
        return self.subconv(x).view(-1).shape[0]


class PretrainNet_T(nn.Module):

    def __init__(
        self,
        in_channel=3,
        sequence_lens=1000,
        time_lens=10,
        hidden_size=64,
        output_size=2,
        layer_size=1,
        bidirectional=True
    ):
        super(PretrainNet_T, self).__init__()

        if sequence_lens % time_lens != 0:
            raise ValueError("Invalid time lens")

        self.in_channel  = in_channel
        self.time_lens   = time_lens
        self.hidden_size = hidden_size
        self.layer_size  = layer_size
        self.window_size = sequence_lens // time_lens
        self.device      = torch.device('cuda')

        self.subconv    = SubConvNet(in_channel=in_channel, out_channel=4)
        self.input_size = self._adaptive_feature_size()

        self.lstm = nn.LSTM(self.input_size, hidden_size, layer_size, bidirectional=bidirectional)
        if bidirectional:
            self.layer_size *= 2

        self.fn1  = nn.Linear(hidden_size * self.layer_size, 128)
        self.fn2  = nn.Linear(128, output_size)

    def forward(self, x):

        batch_size = x.shape[0]

        x = x.chunk(self.time_lens, 2)
        x = torch.stack(x, 1)
        x = x.reshape(batch_size * self.time_lens, self.in_channel, self.window_size)

        x = self.subconv(x)
        x = x.view(batch_size, self.time_lens, self.input_size)
        x = x.permute(1, 0, 2)

        h_0 = torch.zeros(self.layer_size, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.zeros(self.layer_size, batch_size, self.hidden_size).to(self.device)
        x, (h_final, c_final) = self.lstm(x, (h_0, c_0))
        # seq, batch, feature
        x = x.permute(1, 2, 0)

        x = F.avg_pool1d(x, self.time_lens)
        x = x.view(batch_size, -1)
        x = F.relu(self.fn1(x), inplace=True)
        x = F.softmax(self.fn2(x), dim=-1)
        return x

    def _adaptive_feature_size(self):
        x = torch.zeros(1, self.in_channel, self.window_size)
        return self.subconv(x).view(-1).shape[0]


class EncodeNet_T(nn.Module):

    def __init__(
        self,
        in_channel=3,
        sequence_lens=1000,
        time_lens=10,
        hidden_size=64,
        output_size=2,
        layer_size=1,
        bidirectional=True
    ):
        super(EncodeNet_T, self).__init__()

        if sequence_lens % time_lens != 0:
            raise ValueError("Invalid time lens")

        self.in_channel  = in_channel
        self.time_lens   = time_lens
        self.hidden_size = hidden_size
        self.layer_size  = layer_size
        self.window_size = sequence_lens // time_lens
        self.device      = torch.device('cuda')

        self.subconv    = SubConvNet(in_channel=in_channel, out_channel=4)
        self.input_size = self._adaptive_feature_size()

        self.lstm = nn.LSTM(self.input_size, hidden_size, layer_size, bidirectional=bidirectional)
        if bidirectional:
            self.layer_size *= 2

    def forward(self, x):

        batch_size = x.shape[0]

        x = x.chunk(self.time_lens, 2)
        x = torch.stack(x, 1)
        x = x.reshape(batch_size * self.time_lens, self.in_channel, self.window_size)

        x = self.subconv(x)
        x = x.view(batch_size, self.time_lens, self.input_size)
        x = x.permute(1, 0, 2)

        h_0 = torch.zeros(self.layer_size, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.zeros(self.layer_size, batch_size, self.hidden_size).to(self.device)
        x, (h_final, c_final) = self.lstm(x, (h_0, c_0))
        # seq, batch, feature
        x = x.permute(1, 0, 2)
        return x

    def _adaptive_feature_size(self):
        x = torch.zeros(1, self.in_channel, self.window_size)
        return self.subconv(x).view(-1).shape[0]
