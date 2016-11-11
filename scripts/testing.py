import os
import json
import time
import numpy as np




def load_data(data_file):
    csv_delimiter = ';'
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

if __name__ == '__main__':
    path = 'H:\\Thesis_data\\02_inception\\'
    ref_path = 'H:\\00_vanilla\\'
    for f in os.listdir(path):
        if f.endswith(".json"):
            json_file = os.path.join(path, f)
        if f.endswith(".csv"):
            csv_file = os.path.join(path, f)
    for f in os.listdir(ref_path):
        if f.endswith(".json"):
            ref_file = os.path.join(ref_path, f)
        if f.endswith(".csv"):
            csv_ref_file = os.path.join(ref_path, f)

    csv_data = load_data(csv_file)
    csv_ref = load_data(csv_ref_file)
    print get_max_accs(csv_data)
    print get_avg_acc_1(csv_data, 10)
    print get_avg_loss(csv_data, 10)
