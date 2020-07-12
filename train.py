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

BATCH_SIZE = 200
EPOCH = 45
CHANNEL = 8
SIZE =  13

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

X_train, X_test, Y_train, Y_test = load_data.loadFiltData(CHANNEL, pca=False, std=False, size=SIZE)
train_dataset = load_data.MyDataset(X_train, Y_train, device=device)
test_dataset = load_data.MyDataset(X_test, Y_test, device=device)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=40, shuffle=True, drop_last=False)

#net = model.Conv2d(CHANNEL)
net = model.Conv(CHANNEL, BATCH_SIZE)
#net = model.ConvPic(3, BATCH_SIZE)
net.apply(model.weight_init)
net.to(device)

optimizer = optim.SGD(net.parameters(), lr=0.01, weight_decay=5e-4)
#optimizer = optim.Adam(net.parameters(), lr=0.003, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

train_acc_item = []
train_los_item = []
test_loss_item = []
test_accr_item = []
trai_lab1_item = []
trai_alli_item = []
trai_true_item = []
test_lab1_item = []
test_alli_item = []
test_true_item = []

for epoch in range(EPOCH):
    print('\nEpoch {}/{}'.format(epoch+1, EPOCH))
    print('-' * 20)
    
    size = 0.001
    acc = 0
    run_loss = 0
    n1 = 0
    label_1 = 0
    for input,label in train_loader:

        #input = input.view(-1, CHANNEL, 315)    
        label = label.view(-1)

        output = net(input, True, label.shape[0])

        loss = criterion(output, label)
        run_loss += loss.item()

        pred = output.data.max(1, keepdim=True)[1]
        acc += pred.eq(label.data.view_as(pred)).cpu().sum()
        size += output.shape[0]
        n1 += pred.sum()
        label_1 += sum(label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_los_item.append(run_loss)
    train_acc_item.append(float(acc)/size)
    trai_lab1_item.append(n1)
    trai_alli_item.append(size)
    trai_true_item.append(label_1)
    print("Loss : {:.5f} , pred = 1 :{:4}".format(run_loss,n1))
    print("True : {:4}, All : {:4} \nAccuracy rate : {:.4f}".format(acc, size, float(acc)/size))

    val_size = 0.001
    val_acc = 0
    val_loss = 0
    label_1 = 0
    pred_1 = 0
    for input,label in test_loader:

        #input = input.view(-1, CHANNEL, 315)
        label = label.view(-1)

        output = net(input, False, label.shape[0])
        
        pred = output.data.max(1, keepdim=True)[1]
        loss = criterion(output,label)

        val_acc += pred.eq(label.data.view_as(pred)).cpu().sum()
        val_size += output.shape[0]
        val_loss += loss.item()
        label_1 += sum(label)
        pred_1 += pred.sum()

    test_loss_item.append(val_loss)
    test_accr_item.append(float(val_acc)/val_size)
    test_lab1_item.append(pred_1)
    test_alli_item.append(val_size)
    test_true_item.append(label_1)
    print("-"*5)
    print("Val Phase: Acc : {:.4f} , Loss : {:.5f}\n".format(float(val_acc)/val_size, val_loss))

print("\n")
print("*"*50)

# dic = load_data.getTest(CHANNEL)
# acc = 0
# size =  0

# for i,j in dic.items():
#     data,label = j
#     size += 1

#     input = torch.FloatTensor(data).to(device)
#     input = input.view(-1, CHANNEL, 315)

#     output = net(input, False)
#     pred = output.data.max(1, keepdim=True)[1]
    
#     if pred == torch.tensor([[0]]).to(device):
#         pred = 'nonconfused'
#     else:
#         pred = 'confused'
#     if label == pred:
#         acc += 1
#     print("Word:{:15} , Label:{:12} , Prediction:{:12} , Correct ? {:4}, output : {} ".format(i,label,pred,"Yes" if label == pred else "No",output))

# print("-"*50)
# print("Total : {} , Correct : {} , Acc Rate : {:.4f}".format(size, acc, float(acc)/size))
plt.figure(figsize=(16, 9))
plt.subplot(2,2,1)
plt.plot(range(len(train_los_item)), train_los_item, label='train loss')
plt.plot(range(len(test_loss_item)), test_loss_item, label='test loss')
plt.title('Loss')
plt.legend()
plt.subplot(2,2,2)
plt.plot(range(len(train_acc_item)), train_acc_item, label='train acc')
plt.plot(range(len(test_accr_item)), test_accr_item, label='test acc')
plt.title('Accuracy')
plt.legend()
plt.subplot(2,2,3)
plt.plot(range(len(trai_lab1_item)), trai_lab1_item, label="pred 1")
plt.plot(range(len(trai_alli_item)), trai_alli_item, label='all size')
plt.plot(range(len(trai_true_item)), trai_true_item, label='true 1')
plt.title('Train Classification Result')
plt.legend()
plt.subplot(2,2,4)
plt.plot(range(len(test_lab1_item)), test_lab1_item, label="pred 1")
plt.plot(range(len(test_alli_item)), test_alli_item, label='all size')
plt.plot(range(len(test_true_item)), test_true_item, label='true 1')
plt.title('Test Classification Result')
plt.legend()
plt.show()