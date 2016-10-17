import caffe
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import toimage

# 1. load network
network_path = '/home/ellerch/caffeProject/auto_trainer_output/_gute_runs/alexnet_80k/job0/deploy_alexnet_places365.prototxt'
weight_path = '/home/ellerch/caffeProject/auto_trainer_output/_gute_runs/alexnet_80k/job0/_iter_75000.caffemodel'
output_path = '/home/ellerch/caffeProject/auto_trainer_output/_gute_runs/alexnet_80k/'

net = caffe.Net(network_path, weight_path, caffe.TEST)

# 2. get filters
for name, dim in [(k, v[0].data.shape) for k, v in net.params.items()]:
    print '{}, Dims:{}'.format(name, dim)
    # 96 x 11x11 x3
    if 'conv' not in name:
        continue
    filters = net.params[name][0].data
    print np.shape(filters)
    filter_size = filters[0][0][0]
    # 3. compose new image
    comp_im = Image.new("RGB", ((filter_size + 1) * 10, (filter_size + 1) * 10), "white")
    offset_x = 0
    offset_y = 0
    for i in range(np.shape(filters)[0]):
        comp_im.paste(toimage(filters[i]),
                      (offset_x, offset_y))
        offset_x += filter_size + 1
        if offset_x > (filter_size + 1) * 10:
            offset_x = 0
            offset_y += filter_size + 1

    # 4. save to path
    comp_im = comp_im.resize((400, 400), Image.ANTIALIAS)
    comp_im.save(output_path + name + '_filters.png')
#w, h = 512, 512
#data = np.zeros((h, w, 3), dtype=np.uint8)
#data[256, 256] = [255, 0, 0]
#img = Image.fromarray(nice_edge_detectors, 'RGB')
#img.save('my.png')
#img.show()

#print np.shape(nice_edge_detectors.data)
#batch_size, filter_num,x_size,y_size = np.shape(nice_edge_detectors.data)
#compose_image = np.zeros((filter_num/2 * (x_size+2), filter_num/2 * (x_size+2)), dtype=np.uint8)
#x_offset,y_offset = (0,0)
#pixel_vals = 0
#for i in range(filter_num):
#    print 'filter {} starts at {},{}'.format(i,x_offset,y_offset)
#    for x in range(x_size):
#        for y in range(y_size):
#            pixel_vals += nice_edge_detectors.data[0, i, x, y]
#            compose_image[x_offset + x, y_offset + y] = nice_edge_detectors.data[0, i, x, y]
#    x_offset += x_size+1
#    if x_offset + x_size > (filter_num / 2 * (x_size + 2)):
#        x_offset = 0
#        y_offset += y_size + 1
#
#print np.shape(compose_image)
#print pixel_vals
#img = Image.fromarray(compose_image)
#img.save('my.png')
#img.show()
#print higher_level_filter.data

