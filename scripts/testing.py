from PIL import Image
import numpy as np
import math
import subprocess
import os


if __name__ == '__main__':
    script_path = '/home/ellerch/bin/caffe/python/classify.py'
    input_file = '/home/ellerch/caffeProject/scripts/classify_test/jcxdqj4r.bmp'
    output_file = '/home/ellerch/caffeProject/scripts/classify_test/jcxdqj4r.output'
    model_def = '/home/ellerch/caffeProject/web_interface/cnn/sce_vgg_16/sce_deploy_vgg16_places365.prototxt'
    pretrained_model = '/home/ellerch/caffeProject/web_interface/cnn/sce_vgg_16/sce_vgg16_places365.caffemodel'
    images_dim = '256,256'
    mean_file = '/home/ellerch/db/places365/places365CNN_mean.binaryproto'
    input_scale = 1.0
    raw_scale = 255.0,
    channel_swap = '2,1,0'

    print 'plotting learning curve as png'
    # plot_script = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'progress_plot.py')
    process = subprocess.Popen(['python',
                                script_path,
                                input_file,
                                output_file,
                                '--model_def',
                                model_def,
                                '--pretrained_model',
                                pretrained_model,
                                '--gpu',
                                '--images_dim',
                                images_dim,
                                "--mean_file", mean_file,
                                "--input_scale", input_scale,
                                "--raw_scale", raw_scale,
                                "--channel_swap", channel_swap],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    # os.path.join(os.path.dirname(path), 'caffe_log_test{}.csv'.format(os.path.basename(path)[-5]))],
    output = process.communicate()[0]
    print output