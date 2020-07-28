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

net = FrequencyAttentionNet().to(device)
criterion_cel = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)
writer = SummaryWriter("runs/X-FrequencyAttentionNet_feature_withLSTM_" + str(datetime.datetime.now()))

stat = np.zeros(3, dtype=np.float)
test_size = 0
epoch = 30
print('<<=== Begin ===>>')
for i in range(epoch):
    train_correct = train_total = 0
    test_correct  = test_total  = 0
    train_loss = test_loss = 0
    
    net.train()
    for input, label in train_loader:
        #output = net(input)
        output, attn = net(input)
        prediction = torch.argmax(output, 1)
        label = label.view(-1)
        
        loss = criterion_cel(output, label)
        train_loss += loss.item()
        
        train_correct += (prediction == label).sum().float()
        train_total += len(label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
    net.eval()
    for input, label in test_loader:
#         output = net(input)
        output, attn = net(input)
        if i == epoch - 1:
            attn = attn.detach().cpu().numpy()
            test_size += attn.shape[0]
            stat += attn.sum(0)
        
        prediction = torch.argmax(output, 1)
        label = label.view(-1)

        loss = criterion_cel(output, label)
        test_loss += loss.item()

        test_correct += (prediction == label).sum().float()
        test_total += len(label)

    if i % 5 == 0:
        print('e', i)
    writer.add_scalar('loss/train', train_loss, i)
    writer.add_scalar('loss/test', test_loss, i)
    writer.add_scalar('accuracy/train', train_correct/train_total, i)
    writer.add_scalar('accuracy/test', test_correct/test_total, i)
writer.close()
print('<<=== Finish ===>>')