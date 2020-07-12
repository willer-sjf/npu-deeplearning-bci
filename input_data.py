#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
PATH = 'word_EEG_with_label/splitted_word_with_label'
INFORMATION_POS = [3,4,5,6,7,8,9,10]


def getData(filename):
    data = []
    with open(PATH + filename + '.csv') as file:
        for i in file:
            data_line = i.split(',')
            data.append(data_line)
    return data

def getAllData():
    data = []
    data_line = []
    label = []
    for index in range(1,13):
        with open(PATH + '{}.csv'.format(index)) as file:
            data_mat = []
            data_line = []
            for col,line in enumerate(file):
                if col == 0:
                    continue
                line = line.split(',')
                data_line = [float(i) for i in line[3:11]]
                data_mat.append(data_line)
                if col % 315 == 0:
                    data.append(data_mat)
                    label.append([1,0] if line[-2] == 'confused' else [0,1])
                    data_mat = []
    return data,label

def helper():
    data = []
    label = []
    data_mat = []
    label_mat = []
    for i in range(1, 13):

        df = pd.read_csv(PATH + "{}.csv" .format(i))
        array = df.values
        for i,line in enumerate(array):

            if i % 315 == 0 and i != 0:
                data.append(data_mat)
                data_mat = []
        data.append(array[:, 3: 11])

        tmp_label = np.zeros((array.shape[0], ))
        for index in range(array.shape[0]):
            if array[i, -2] == "nonconfused":
                tmp_label[index] = 1
        label.append(tmp_label)

def saveMatrix(data,i):
    s = set()
    mat = []
    confused = 0
    nonconfused = 0
    for line in data:
        if line[15] in s:
            continue
        else:
            mat.append([i,line[14],line[15]])
            if line[14].strip() == "confused":
                confused += 1
            else:
                nonconfused += 1
            s.add(line[15])
    print(confused,nonconfused)
    mat[0] = [0,"标记","单词","confused数量：",confused,"nonconfused数量:",nonconfused]
    frame = pd.DataFrame(mat)
    frame.to_csv('summary{}.csv'.format(i))
    return confused,nonconfused

def getLineData():
    data = []
    label = []
    for pos in range(1,13):
        with open(PATH + '{}.csv'.format(pos)) as file:
            for index,line in enumerate(file):
                if index == 0:
                    continue
                data_line = line.split(',')
                data_line = [float(i) for i in line[3:11]]
                data.append(data_line)
                label.append(1 if line[-2] == 'confused' else 0)
    return data,label

if __name__=="__main__":
    d,l = getAllData()
    d = np.array(d)
    l = np.array(l)
    print(d.shape,d.ndim)
    print(l.shape)




