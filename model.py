import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ResidualBlock1d(nn.Module):
    def __init__(self,in_channel, out_channel, kernel_size=3, stride=1, padding=1,dilation=1):
        super(ResidualBlock1d, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(in_channel),
            nn.ReLU(),
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
    
    def __init__(self):
        super(PureCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=40 , out_channels=128, kernel_size=8, stride=4)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=4, stride=2)
        
        self.max_pool1 = nn.MaxPool1d(kernel_size=2)
        self.max_pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.block1 = ResidualBlock1d(128, 128, 3, padding=1)
        self.block2 = ResidualBlock1d(128, 128, 3, padding=1)
        self.block3 = ResidualBlock1d(128, 128, 3, padding=1)
        self.block4 = nn.Conv1d(in_channels=128, out_channels=8, kernel_size=3, stride=1)
        
        self.fn1 = nn.Linear(1992, 1024)
        self.fn2 = nn.Linear(1024, 2)
        
        self.bn1 = nn.BatchNorm1d(40)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.drop1 = nn.Dropout(0.3)
        self.drop2 = nn.Dropout(0.3)
        self.drop3 = nn.Dropout(0.3)
        
    def forward(self, x):
        
        #x = x[:, :32, 2048:]
        x = self.bn1(x)
        x = self.conv1(x)
        x = F.relu(self.bn2(x))
        x = self.max_pool1(x)
        
        x = self.bn2(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        x = x.view(-1, 1992)
        x = F.relu(self.fn1(x))
        x = F.softmax(self.fn2(x), -1)
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
    
    def __init__(self, in_channel, hidden_channel=64, out_channel=2):
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
    
class AttentionAwareNet(nn.Module):
    
    def __init__(self, hidden_size):
        super(AttentionAwareNet, self).__init__()
        self.q_ = nn.Linear()
        
    def forward(self, x):
        
        
        attn = F.softmax(score, -1)
        out = torch.matmul(attn, x)
        return out
    
    
class AttentionNet(nn.Module):
    
    def __init__(self, 
                 time_lens=63, 
                 input_size=126, 
                 hidden_size=256, 
                 output_size=2, 
                 layer_size=1, 
                 batch_size=64,
                 bidirectional=False
                ):
        super(AttentionNet, self).__init__()
        
        self.time_lens = time_lens
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.layer_size  = layer_size
        self.batch_size  = batch_size
        self.device = torch.device('cuda')
        
        self.subconv = SubConvNet(in_channel=40)
        
        self.lstm = nn.LSTM(input_size, hidden_size, layer_size, bidirectional=bidirectional)
        #self.attn = AttentionAwareNet()
        self.maxpool = nn.MaxPool1d(time_lens)
        self.fn = nn.Linear(hidden_size * layer_size, output_size)

    def forward(self, x):
        
        x = x.chunk(self.time_lens, 2)
        x = torch.stack(x, 1)
        x = x.reshape(self.batch_size * self.time_lens, 40, 128)
        x = self.subconv(x)
        
        x = x.view(self.batch_size, self.time_lens, self.input_size)
        x = x.permute(1, 0, 2)
        
        h_0 = torch.zeros(self.layer_size, self.batch_size, self.hidden_size).to(self.device)
        c_0 = torch.zeros(self.layer_size, self.batch_size, self.hidden_size).to(self.device)
        
        x, (h_final, c_final) = self.lstm(x, (h_0, c_0))
        x = x.permute(1, 2, 0)
        x = self.maxpool(x)
        x = x.view(self.batch_size, -1)
        x = F.softmax(self.fn(x), -1)
        
        return x
    
class AttentionFeatureAwareNet(nn.Module):
    
    def __init__(self,
                hidden_size,
                out_size):
        
        self.cnet = SubConvNet()
        self.attn = AttentionAwareNet()
        self.fn = nn.Linear(hidden_size, out_size)
        
    def forward(self, x):
        
        x = x.view(self.batch_size * self.channel_size, -1)
        x = self.cnet(x)
        x = x.view(self.batch_size, self.channel_size, -1)
        x, score = self.attn(x)
        x = self.fn(x)
        return x, score

    
    
class ScaledDotProductAttention(nn.Module): # 点乘
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask): # 实现注意力公式
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module): # 多头注意力
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
   
    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        output = nn.Linear(n_heads * d_v, d_model)(context)
        return nn.LayerNorm(d_model)(output + residual), attn
    
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

    def forward(self, inputs):
        residual = inputs # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return nn.LayerNorm(d_model)(output + residual)