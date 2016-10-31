# https://gist.github.com/yassersouri/f617bf7eff9172290b4f
import caffe
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pylab as plt
import pylab

net = caffe.Classifier('/path/to/caffe/models/bvlc_reference_caffenet/deploy.prototxt',
                       '/path/to/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                       channel_swap=(2, 1, 0), raw_scale=255)
net_mean = caffe.Classifier('/path/to/caffe/models/bvlc_reference_caffenet/deploy.prototxt',
                            '/path/to/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                            mean=np.load('/path/to/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'),
                            channel_swap=(2, 1, 0), raw_scale=255)

fake = np.ones((227, 227, 3))

fake_pre = net_mean.preprocess('data', fake)
fake_re = net.deprocess('data', fake_pre)

mean_image = 1 - fake_re

plt.imshow(mean_image)
