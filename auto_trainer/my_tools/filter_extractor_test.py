import caffe
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import toimage

network_path = '/home/ellerch/caffeProject/auto_trainer_output/_gute_runs/alexnet_80k/job0/deploy_alexnet_places365.prototxt'
weight_path = '/home/ellerch/caffeProject/auto_trainer_output/_gute_runs/alexnet_80k/job0/_iter_75000.caffemodel'
net = caffe.Net(network_path, weight_path, caffe.TEST)

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

nice_edge_detectors = net.blobs['conv1']
higher_level_filter = net.blobs['fc7']
print [(k, v[0].data.shape) for k, v in net.params.items()]
# the parameters are a list of [weights, biases]
filters = net.params['conv1'][0].data
print np.shape(filters)
#vis_square(filters.transpose(0, 2, 3, 1))
#feat = net.blobs['conv1'].data[4, :36]
#vis_square(feat, padval=1)
path = '/home/ellerch/caffeProject/auto_trainer_output/_gute_runs/alexnet_80k/con1_filters/'
for i in range(np.shape(filters)[0]):
    toimage(filters[i]).save(path+'{}.png'.format(i))

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

