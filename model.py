import torch
import torch.nn as nn
import torch.nn.functional as F

class PureCNN(nn.Module):
    
    def __init__(self):
        super(PureCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=40, out_channels=64, kernel_size=8, stride=4)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=8, stride=4)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=4, stride=2)
        
        self.max_pool1 = nn.MaxPool1d(kernel_size=4)
        self.max_pool2 = nn.MaxPool1d(kernel_size=2)
        self.max_pool3 = nn.MaxPool1d(kernel_size=2)
        
        self.fn1 = nn.Linear(960, 512)
        self.fn2 = nn.Linear(512, 2)
        
        self.bn1 = nn.BatchNorm1d(40)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        
        x = self.bn1(x)
        x = self.conv1(x)
        x = F.relu(self.bn2(x))
        x = self.max_pool1(x)
        
        x = self.conv2(x)
        x = F.relu(self.bn3(x))
        x = self.max_pool2(x)
        
        x = self.conv3(x)
        x = F.relu(self.bn4(x))
        x = self.max_pool3(x)
        
        x = x.view(-1)
        x = F.relu(self.fn1(x))
        x = F.softmax(self.fn2(x), -1)
        return x
    
class DirectLSTM(nn.Module):
    
    def __init__(self):
        super(DirectLSTM, self).__init__()
        
    def forward(self, x):
        pass
    
class ConvLSTM(nn.Module):
    
    def __init__(self):
        super(ConvLSTM, self).__init__()
        
    def forward(self, x):
        pass
        
class AttentionNet(nn.Module):
    
    def __init__(self):
        super(AttentionNet, self).__init__()
       
    def forward(self, x):
        pass