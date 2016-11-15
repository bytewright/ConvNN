from PIL import Image
import numpy as np
import math
import subprocess
import os
import numpy as np
import sys


csv_delimiter = ';'
iter_index = 0
acc_index = 1
loss_index = 3


def load_data(data_file):
    #delimiter = ';'
    test_data = []
    with open(data_file, 'r') as f:
        for line in f.readlines():
            #print line
            line = line.rstrip()
            if line.split(csv_delimiter).__len__() < 3:
                print('incomplete line found')
            else:
                test_data.append([data for data in line.split(csv_delimiter) if data is not ''])

    return test_data[1:]


def get_max_accs(data):
    acc1 = [float(x[1]) for x in data]
    acc5 = [float(x[2]) for x in data]
    return (max(acc1))*100, (max(acc5))*100


def get_avg_acc_1(data, num_items):
    acc1 = [x[1] for x in data]
    sum = 0.0
    items = 0.0
    for acc in acc1[-num_items:]:
        sum += float(acc)
        items += 1.0
    return (sum/items)*100


def get_avg_acc_5(data, num_items):
    acc5 = [x[2] for x in data]
    sum = 0.0
    items = 0.0
    for acc in acc5[-num_items:]:
        sum += float(acc)
        items += 1.0
    return (sum/items)*100


def get_avg_loss(data, num_items):
    loss = [x[3] for x in data]
    sum = 0.0
    items = 0.0
    for acc in loss[-num_items:]:
        sum += float(acc)
        items += 1.0
    return sum / items


def get_training_stats(csv_path):
    csv_delimiter = ';'
    acc1_index = 1
    acc5_index = 2
    loss_index = 3
    test_data = []
    with open(csv_path, 'r') as f:
        for line in f.readlines():
            # print line
            line = line.rstrip()
            test_data.append([data for data in line.split(csv_delimiter) if data is not ''])

    csv_data = test_data[1:]
    num_items = 10

    acc1 = [float(x[acc1_index]) for x in reversed(csv_data)]
    acc5 = [float(x[acc5_index]) for x in reversed(csv_data)]
    loss = [float(x[loss_index]) for x in reversed(csv_data)]
    max_top1 = (max(acc1))
    max_top5 = (max(acc5))
    avg_loss = 0.0
    avg_acc_top1 = 0.0
    avg_acc_top5 = 0.0

    for i in range(num_items):
        avg_acc_top1 += acc1[i] / num_items
        avg_acc_top5 += acc5[i] / num_items
        avg_loss += loss[i] / num_items

    stats = {'max_top1': round(max_top1*100, 3),
             'max_top5': round(max_top5*100, 3),
             'avg_loss': round(avg_loss, 3),
             'avg_acc_top1': round(avg_acc_top1*100, 3),
             'avg_acc_top5': round(avg_acc_top5*100, 3)}
    return stats

if __name__ == '__main__':
    csv_path = os.path.normpath("H:/Thesis DVD/Thesis_experimente/00_Referenznetz/logs/thesis_exp0_vanilla_caffe_test_log.csv")

    print get_training_stats(csv_path)


