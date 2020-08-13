import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class ResidualBlock1d(nn.Module):
    def __init__(self, inner_channel, kernel_size=3, stride=1, padding=1, dilation=1):
        super(ResidualBlock1d, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(inner_channel),
            nn.ReLU(),
            nn.Conv1d(inner_channel, inner_channel, kernel_size, stride, padding, dilation, bias=False),
            nn.BatchNorm1d(inner_channel),
            nn.ReLU(),
            nn.Conv1d(inner_channel, inner_channel, kernel_size, stride, padding, dilation, bias=False),
        )

    def forward(self, x):
        out = self.block(x)
        out += x
        return out

class SubConvNet(nn.Module):

    def __init__(self, in_channel=1, out_channel=4, hidden_channel=64):
        super(SubConvNet, self).__init__()

        self.conv = nn.Conv1d(in_channel, hidden_channel, kernel_size=4, stride=2)
        self.block1 = ResidualBlock1d(hidden_channel)
        self.block2 = ResidualBlock1d(hidden_channel)
        self.conb = nn.Conv1d(hidden_channel, out_channel, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.conb(x)
        x = x.view(batch_size, -1)
        return x


class SubFeatureNet(nn.Module):
    def __init__(
        self,
        sequence_lens=1000,
        time_lens=10,
        hidden_size=256,
        layer_size=1,
        feature_size=256,
        bidirectional=False
    ):
        super(SubFeatureNet, self).__init__()

        self.time_lens   = time_lens
        self.hidden_size = hidden_size
        self.layer_size  = layer_size
        self.window_size = sequence_lens // time_lens
        self.device      = torch.device('cuda')

        self.subconv = SubConvNet(in_channel=1, out_channel=4)
        self.input_size = self._adaptive_feature_size()

        self.lstm = nn.LSTM(self.input_size, hidden_size, layer_size, bidirectional=bidirectional)
        if bidirectional:
            self.layer_size *= 2
        self.transform = nn.Linear(hidden_size * self.layer_size, feature_size)


    def forward(self, x):

        batch_size = x.shape[0]

        x = x.chunk(self.time_lens, 2)
        x = torch.stack(x, 1)
        x = x.reshape(batch_size * self.time_lens, 1, self.window_size)

        x = self.subconv(x)
        x = x.view(batch_size, self.time_lens, self.input_size)
        x = x.permute(1, 0, 2)

        h_0 = torch.zeros(self.layer_size, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.zeros(self.layer_size, batch_size, self.hidden_size).to(self.device)
        x, (h_final, c_final) = self.lstm(x, (h_0, c_0))
        x = x[-1, :, :]

        x = x.view(batch_size, -1)
        x = torch.tanh(self.transform(x))
        return x

    def _adaptive_feature_size(self):
        x = torch.zeros(1, 1, self.window_size)
        return self.subconv(x).view(-1).shape[0]


class PureCNN(nn.Module):

    def __init__(
        self,
        in_channel=3,
        sequence_lens=1000,
        residual_channel=128,
        fully_unit=512,
        output_size=2,
    ):
        super(PureCNN, self).__init__()
        self.in_channel    = in_channel
        self.sequence_lens = sequence_lens

        self.conv1 = nn.Conv1d(in_channels=in_channel , out_channels=residual_channel, kernel_size=4, stride=2)
        self.conv2 = nn.Conv1d(in_channels=residual_channel, out_channels=residual_channel, kernel_size=4, stride=2)

        self.block1 = ResidualBlock1d(residual_channel, 3, padding=1)
        self.block2 = ResidualBlock1d(residual_channel, 3, padding=1)
        self.block3 = ResidualBlock1d(residual_channel, 3, padding=1)
        self.block4 = nn.Conv1d(in_channels=residual_channel, out_channels=2, kernel_size=3, stride=1)

        self.feature_size = self._adaptive_feature_size()
        self.fn1 = nn.Linear(self.feature_size, fully_unit)
        self.fn2 = nn.Linear(fully_unit, output_size)

        self.bnf = nn.BatchNorm1d(in_channel)
        self.bn1 = nn.BatchNorm1d(residual_channel)


    def forward(self, x):

        x = self.bnf(x)

        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = x.view(-1, self.feature_size)
        x = F.relu(self.fn1(x))
        x = F.softmax(self.fn2(x), -1)
        return x

    def _adaptive_feature_size(self):
        x = torch.zeros(1, self.in_channel, self.sequence_lens)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.view(-1).shape[0]


class ConvLSTM(nn.Module):

    def __init__(
        self,
        in_channel=3,
        sequence_lens=1000,
        time_lens=10,
        hidden_size=256,
        output_size=2,
        layer_size=1,
        bidirectional=False
    ):
        super(ConvLSTM, self).__init__()

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
        self.fn = nn.Linear(hidden_size * self.layer_size, output_size)


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
        x = x[-1, :, :]
        x = F.softmax(self.fn(x), -1)
        return x

    def _adaptive_feature_size(self):
        x = torch.zeros(1, self.in_channel, self.window_size)
        return self.subconv(x).view(-1).shape[0]


class TemporalAttentionNet(nn.Module):

    def __init__(
        self,
        in_channel=3,
        sequence_lens=1000,
        time_lens=10,
        hidden_size=256,
        hidden_attention_size=128,
        output_size=2,
        layer_size=1,
        bidirectional=True
    ):
        super(TemporalAttentionNet, self).__init__()

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
        self.attn = AttentionFeatureNet(hidden_size * self.layer_size, hidden_attention_size)
        self.fn   = nn.Linear(hidden_size * self.layer_size, output_size)


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
        x = x.permute(1, 0, 2)

        x, attn = self.attn(x)
        x = self.fn(x)
        x = F.softmax(x, dim=-1)
        return x, attn

    def _adaptive_feature_size(self):
        x = torch.zeros(1, self.in_channel, self.window_size)
        return self.subconv(x).view(-1).shape[0]


class SpacialAttentionNet(nn.Module):
    def __init__(
        self,
        in_channel=3,
        sequence_lens=1000,
        time_lens=10,
        hidden_size=256,
        hidden_attention_size=512,
        output_size=2,
        layer_size=1,
        bidirectional=True
    ):
        super(SpacialAttentionNet, self).__init__()

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

        self.attn = AttentionFeatureNet(hidden_size * self.layer_size, hidden_attention_size)
        self.fn = nn.Linear(hidden_size * self.layer_size, output_size)

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
        x, attn = self.attn(x)
        x = self.fn(x)
        x = F.softmax(x, dim=-1)
        return x, attn

    def _adaptive_feature_size(self):
        x = torch.zeros(1, 1, self.window_size)
        return self.subconv(x).view(-1).shape[0]


class FrequencyAttentionNet(nn.Module):

    def __init__(
        self,
        in_channel=3,
        sequence_lens=1000,
        time_lens=10,
        output_size=2,
        feature_size=256,
        band_size=3,
    ):
        super(FrequencyAttentionNet, self).__init__()
        if in_channel % band_size != 0:
            raise ValueError("Invalid input channel")
        if sequence_lens % time_lens != 0:
            raise ValueError("Invalid time lens")

        self.in_channel    = in_channel
        self.sequence_lens = sequence_lens
        self.per_channel   = in_channel // band_size
        self.feature_size  = feature_size
        self.band_size     = band_size

        self.band_net = nn.ModuleList()
        for i in range(band_size):
            sub_net = SubFeatureNet(sequence_lens=sequence_lens,
                                    time_lens=time_lens,
                                    hidden_size=256,
                                    layer_size=1,
                                    feature_size=feature_size,
                                    bidirectional=True)
            self.band_net.append(sub_net)

        self.attn = AttentionFeatureNet(self.feature_size)
        self.fn = nn.Linear(self.feature_size, output_size)

    def forward(self, x):

        batch_size = x.shape[0]

        x = x.chunk(self.band_size, 1)
        x = torch.stack([self.band_net[i](x[i].reshape(batch_size*self.per_channel, 1, -1)) for i in range(self.band_size)], 1)
        x = x.view(batch_size, self.in_channel, -1)

        x, attn = self.attn(x)
        x = self.fn(x)
        x = F.softmax(x, dim=-1)
        return x, attn


class AttentionFeatureNet(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=256,
    ):
        super(AttentionFeatureNet, self).__init__()
        self.score1 = nn.Linear(input_size, hidden_size)
        self.score2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        channel_size = x.shape[1]

        attn = F.relu(self.score1(x))
        attn = self.score2(attn)
        attn = F.softmax(attn, 1)
        out = torch.sum(attn * x, dim=1)
        return out, attn.view(batch_size, -1)
