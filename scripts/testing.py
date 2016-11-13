from PIL import Image
import numpy as np
import math
import subprocess
import os
import caffe
import numpy as np
import sys

if __name__ == '__main__':
    script_path = '/home/ellerch/bin/caffe/python/classify.py'
    input_file = '/home/ellerch/caffeProject/scripts/classify_test/pzjxnad5.png'
    output_file = '/home/ellerch/caffeProject/scripts/classify_test/pzjxnad5'
    model_def = '/home/ellerch/caffeProject/web_interface/cnn/sce_vgg_16/sce_deploy_vgg16_places365.prototxt'
    pretrained_model = '/home/ellerch/caffeProject/web_interface/cnn/sce_vgg_16/sce_vgg16_places365.caffemodel'
    images_dim = '256,256'
    mean_file = '/home/ellerch/db/places365/places365CNN_mean.binaryproto'
    mean_out_file = '/home/ellerch/db/places365/places365CNN_mean.npy'
    input_scale = "1.0"
    raw_scale = "255.0",
    channel_swap = '2,1,0'


    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(mean_file, 'rb').read()
    blob.ParseFromString(data)
    arr = np.array(caffe.io.blobproto_to_array(blob))
    out = arr[0]
    np.save(mean_out_file, out)
    if False:

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

        lables_path = '/home/ellerch/db/places365/categories_places365.txt'
        labels = []
        with open(lables_path, 'r') as f:
            for line in f.readlines():
                line.rstrip()
                line = line.replace('\n', '')
                labels.append((int(line.split(' ')[1]), line.split(' ')[0]))
        print labels[:10]
        raw_vals = np.load(output_file+'.npy')
        vals = zip([x for x in raw_vals[0]], range(365))

        vals = sorted(vals, reverse=True)
        for val in vals[:10]:
            print '{}:{}'.format(val, labels[val[1]])