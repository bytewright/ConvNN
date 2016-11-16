from PIL import Image
import numpy as np
import math
import subprocess
import os
import numpy as np
import sys

if __name__ == '__main__':
    labels_path = '/home/ellerch/caffeProject/web_interface/cnn/sce_vgg_16/categories_places365.txt'
    labels_path_new = '/home/ellerch/caffeProject/web_interface/cnn/sce_vgg_16/my_categories_places365.txt'

    labels = []
    new_file = ''
    with open(labels_path, 'r') as f:
        for line in f.readlines():
            line.rstrip()
            line = line.replace('\n', '')
            new_file += '{};{}\r\n'.format(line.split(';')[1],line.split(';')[0])
    with open(labels_path_new,'w') as f:
        f.writelines(new_file)