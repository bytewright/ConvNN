import caffe
from PIL import Image
import numpy as np
from scipy.misc import toimage
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='extracts convolution filter from trained model as pngs')
    parser.add_argument('prototxt', help='prototxt of network')
    parser.add_argument('caffemodel', help='trained weights of network')
    parser.add_argument('output_path', help='path to png, will be overwritten by this script')
    args = parser.parse_args()

    # 1. load network
    network_path = args.prototxt
    weight_path = args.caffemodel
    output_path = args.output_path
    #network_path = '/home/ellerch/caffeProject/auto_trainer_output/_gute_runs/my_conv3+4+5_to_1/job7/job7/my_conv3+4+5to1_alexnet.prototxt'
    #weight_path = '/home/ellerch/caffeProject/auto_trainer_output/_gute_runs/my_conv3+4+5_to_1/job7/job7/_iter_75000.caffemodel'
    #output_path = '/home/ellerch/caffeProject/auto_trainer_output/_gute_runs/my_conv3+4+5_to_1/job7/'

    net = caffe.Net(network_path, weight_path, caffe.TEST)

    # 2. get filters
    for name, dim in [(k, v[0].data.shape) for k, v in net.params.items()]:
        print '{}, Dims:{}'.format(name, dim)
        # 96 x 11x11 x3
        if 'conv' not in name:
            print 'layer "{}" has no conv in name, skipping'.format(name)
            continue
        filters = net.params[name][0].data
        print np.shape(filters)
        filter_count, filter_channels, filter_size,_  = np.shape(filters)
        print '{} filters with size: {}x{}x{}'.format(filter_count, filter_size, filter_size, filter_channels)
        if filter_channels is 3:
            # 3. compose new image
            # new size = 120x120
            comp_im = Image.new("RGB", ((filter_size + 1) * 10 - 1, (filter_size + 1) * 10 - 1), "white")
            offset_x = 0
            offset_y = 0
            for i in range(filter_count):
                comp_im.paste(toimage(filters[i]),
                              (offset_x, offset_y))
                offset_x += filter_size + 1
                if offset_x > (filter_size + 1) * 10 - 1:
                    offset_x = 0
                    offset_y += filter_size + 1
            comp_im_big = comp_im.resize((400, 400))
            comp_im_big.save(os.path.join(output_path, name + '_filters_big.png'))
        else:
            comp_im = Image.new("L", (filter_channels * (filter_size + 1), (filter_size + 1) * filter_count), "white")
            offset_x = 0
            offset_y = 0

            for i in range(filter_count):
                for j in range(filter_channels):
                    comp_im.paste(toimage(filters[i][j]),
                                  (offset_x, offset_y))
                    offset_x += filter_size + 1
                    if offset_x > (filter_size + 1) * filter_channels:
                        offset_x = 0
                        offset_y += filter_size + 1
        # 4. save to path
        comp_im.save(os.path.join(output_path, name + '_filters.png'))
