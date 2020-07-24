import load_data
import model

import time
import torch
import random
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
np.set_printoptions(threshold = 1e6)
import datetime
from tensorboardX import SummaryWriter

net = PureCNN().to(device)
#net = AttentionNet().to(device)
criterion_mse = nn.MSELoss()
criterion_cel = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=2e-2)
writer = SummaryWriter("runs/PureCNN_nonorm_stft" + str(datetime.datetime.now()))

print('<<=== Begin ===>>')
for i in range(50):
    train_correct = train_total = 0
    test_correct  = test_total  = 0
    train_loss = test_loss = 0
    
    net.train()
    for input, label in train_loader:

        output = net(input)
        prediction = torch.argmax(output, 1)
        label = torch.argmax(label, 1)
        
        loss = criterion_cel(output, label)
        train_loss += loss.item()
        
        train_correct += (prediction == label).sum().float()
        train_total += len(label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    net.eval()
    with torch.no_grad():
        for input, label in test_loader:
            output = net(input)
            prediction = torch.argmax(output, 1)
            label = torch.argmax(label, 1)
            
            loss = criterion_cel(output, label)
            test_loss += loss.item()
        
            test_correct += (prediction == label).sum().float()
            test_total += len(label)
        
    writer.add_scalar('loss/train', train_loss, i)
    writer.add_scalar('loss/test', test_loss, i)
    writer.add_scalar('accuracy/train', train_correct/train_total, i)
    writer.add_scalar('accuracy/test', test_correct /test_total, i)
writer.close()
print('<<=== Finish ===>>')