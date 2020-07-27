import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class SelectAgent:
    def __init__(self):
        pass
    def select_action(self):
        pass
    def train(self):
        pass
    
class ResidualBlock1d(nn.Module):
    def __init__(self,in_channel, out_channel, kernel_size=3, stride=1, padding=1,dilation=1):
        super(ResidualBlock1d, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(in_channel),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(in_channel, out_channel, kernel_size, stride, padding, dilation),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),
            nn.Conv1d(out_channel,out_channel, kernel_size, stride, padding, dilation),
        )

    def forward(self, x):
        out = self.block(x)
        out += x
        return out

class ResidualBlock2d(nn.Module):
    def __init__(self,in_channel, out_channel, kernel_size=3, stride=1, padding=1,dilation=1):
        super(ResidualBlock2d, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, dilation, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv1d(out_channel,out_channel, kernel_size, stride, padding, dilation, bias=False),
        )

    def forward(self, x):
        out = self.block(x)
        out += x
        return out
        
class PureCNN(nn.Module):
    
    def __init__(self, residual_channel=64, fully_unit=256):
        super(PureCNN, self).__init__()
        
        self.feature_size = 236
        
        self.conv1 = nn.Conv1d(in_channels=3 , out_channels=residual_channel, kernel_size=4, stride=2)
        self.conv2 = nn.Conv1d(in_channels=residual_channel, out_channels=residual_channel, kernel_size=4, stride=2)
        
        self.block1 = ResidualBlock1d(residual_channel, residual_channel, 3, padding=1)
        self.block2 = ResidualBlock1d(residual_channel, residual_channel, 3, padding=1)
        self.block3 = ResidualBlock1d(residual_channel, residual_channel, 3, padding=1)
        
        self.block4 = nn.Conv1d(in_channels=residual_channel, out_channels=4, kernel_size=3, stride=1)
        
 
        self.fn1 = nn.Linear(self.feature_size, fully_unit)
        self.fn2 = nn.Linear(fully_unit, 2)
        

#         self.bn1 = nn.BatchNorm1d(40)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.bn3 = nn.BatchNorm1d(128)
        
#         self.drop1 = nn.Dropout(0.3)
#         self.drop2 = nn.Dropout(0.3)
#         self.drop3 = nn.Dropout(0.3)
        
    def forward(self, x):
        
        # x = self.bn1(x)
        
        x = self.conv1(x)
        x = F.max_pool1d(x, 2)
        
        x = self.conv2(x)
        x = F.max_pool1d(x, 2)
#         x = self.bn2(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = self.max_pool2(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block4(x)
        
        x = x.view(-1, self.feature_size)
        x = F.relu(self.fn1(x))
        x = F.softmax(self.fn2(x), -1)
        return x

class PureLSTM(nn.Module):
    
    def __init__(self,
                time_lens=63,
                pool_stride=8):
        
        super(PureLSTM, self).__init__()
        
        self.feature = nn.Linear(160, 512)
        self.LSTM = nn.LSTM(512, 512)
        self.fn = nn.Linear(512, 2)
        
        self.device = torch.device('cuda')
        
    def forward(self, x):
        
        batch_size = x.size(0)
        x = x.permute(0, 2, 1)
        x = x.view(-1, 160)
        x = F.tanh(self.feature(x))
        x = x.view(batch_size, -1, 512)
        
        h_0 = torch.zeros(1, batch_size, 512).to(self.device)
        c_0 = torch.zeros(1, batch_size, 512).to(self.device)
        
        x = x.permute(2, 0, 1)
        x, (h_final, c_final) = self.LSTM(x, (h_0, c_0))
        h_final = h_final.squeeze()
        x = F.softmax(self.fn(h_final), -1)
        return x
        
class ConvLSTM(nn.Module):
    
    def __init__(self):
        super(ConvLSTM, self).__init__()
        
        self.lstm = nn.LSTM(40, 512, 1)
        self.fn1 = nn.Linear(hidden_size, 512)
        self.fn2 = nn.Linear(512, 2)
        
    def forward(self, x):
        
        h_0 = torch.zeros(1, x, 512)
        c_0 = torch.zeros(1, x, 512)
        
        out = self.lstm(x,(h_0, c_0))
        out = F.relu(self.fn1(out))
        out = self.fn2(out)
        return out
    
        
class SubConvNet(nn.Module):
    
    def __init__(self, in_channel, hidden_channel=32, out_channel=2):
        super(SubConvNet, self).__init__()
        
        self.conv = nn.Conv1d(in_channel, hidden_channel, kernel_size=4, stride=2)
        self.block1 = ResidualBlock1d(hidden_channel, hidden_channel)
        self.block2 = ResidualBlock1d(hidden_channel, hidden_channel)
        self.conb = nn.Conv1d(hidden_channel, out_channel, 1)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.conb(x)
        x = x.view(batch_size, -1)
        return x

class SpacialAttentionNet(nn.Module):
    def __init__(self,):
        super(SpacialAttentionNet, self).__init__()
        
        self.single = SubConvNet(in_channel=1)
        self.attn = AttentionAwareNet()
        
        self.fn = nn.Linear()
        
    def forward(self, x):

        batch_size = x.shape[0]
        channel_size = x.shape[1]
        x = x.view(batch_size * channel_size, -1)
        x = self.single(x)
        x = x.view(batch_size, channel_size, -1)
        x, attn = self.attn(x)
        x = self.fn(x)
        x = F.softmax(x, dim=-1)
        return x, attn

    
class FrequencyAttentionNet(nn.Module):
    def __init__(self,):
        super(SpacialAttentionNet, self).__init__()
        
        self.alpha = SubConvNet()
        self.beta = SubConvNet()
        self.gamma = SubConvNet()
        
        self.alpha_transform = nn.Linear()
        self.beta_transform = nn.Linear()
        self.gamma_transform = nn.Linear()
        
        self.attn = AttentionAwareNet()
        self.fn = nn.Linear()
        
    def forward(self, x):
        batch_size = x.shape[0]
        channel_size = x.shape[1]
        
        alpha = x[:, :3]
        x_a = self.alpha(alpha)
        x_a = self.alpha_transform(x_a)
        
        x_b = self.beta(beta)
        x_b = self.alpha_transform(x_a)
        
        x, attn = self.AttentionAwareNet()
        x = self.fn(x)
        x = F.softmax(x, dim=-1)
        return x, attn
    

class TemporalAttentionNet(nn.Module):
    
    def __init__(self, 
                 time_lens=10, 
                 input_size=98, 
                 hidden_size=256, 
                 output_size=2, 
                 layer_size=1, 
                 bidirectional=False
                ):
        super(TemporalAttentionNet, self).__init__()
        
        self.time_lens = time_lens
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.layer_size  = layer_size
        self.device = torch.device('cuda')
        
        self.subconv = SubConvNet(in_channel=3)
        
        self.lstm = nn.LSTM(input_size, hidden_size, layer_size, bidirectional=bidirectional)
        #self.attn = AttentionAwareNet()
        self.maxpool = nn.MaxPool1d(time_lens)
        self.fn = nn.Linear(hidden_size * layer_size, output_size)

    def forward(self, x):
        
        batch_size = x.shape[0]
        x = x.chunk(self.time_lens, 2)
        x = torch.stack(x, 1)
        x = x.reshape(batch_size * self.time_lens, 3, 100)
        x = self.subconv(x)
        
        x = x.view(batch_size, self.time_lens, self.input_size)
        x = x.permute(1, 0, 2)
        
        h_0 = torch.zeros(self.layer_size, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.zeros(self.layer_size, batch_size, self.hidden_size).to(self.device)
        
        x, (h_final, c_final) = self.lstm(x, (h_0, c_0))
        x = x.permute(1, 2, 0)
        x = self.maxpool(x)
        x = x.view(batch_size, -1)
        x = F.softmax(self.fn(x), -1)
        
        return x
    
class AttentionFeatureAwareNet(nn.Module):
    
    def __init__(self,
                hidden_size,
                out_size):
        
        self.cnet = SubConvNet(in_channel=1, hidden_channel=16, out_channel=2)
        self.attn = AttentionAwareNet(hidden_size)
        self.fn = nn.Linear(hidden_size, out_size)
        
    def forward(self, x):
        
        x = x.view(self.batch_size * self.channel_size, -1)
        x = self.cnet(x)
        x = x.view(self.batch_size, self.channel_size, -1)
        x, score = self.attn(x)
        x = self.fn(x)
        return x, score
        
    
class AttentionAwareNet(nn.Module):
    def __init__(self, input_size=64, hidden_size=64):
        super(AttentionAwareNet, self).__init__()
        
        self.K = nn.Linear(input_size, hidden_size)
        self.V = nn.Linear(input_size, hidden_size)
        self.Q = torch.randn(hidden_size, requires_grad=False)
   
    def forward(self, x):
        
        k_s = self.K(x)
        v_s = self.V(x)
        attn = torch.matmul(self.Q, k_s.transpose(-1, -2))
        print(attn.shape)
        attn = F.softmax(attn, -1)
        print(attn.shape, v_s.shape)
        output = torch.matmul(attn, v_s)
        return output, attn

    