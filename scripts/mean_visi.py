# https://gist.github.com/yassersouri/f617bf7eff9172290b4f
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pylab as plt
import caffe
import numpy as np
import pylab


model_path = '/home/ellerch/caffeProject/auto_trainer_output/2016-10-30_11h-46m-18s_experiment/vanilla_alexnet/deploy_alexnet_places365.prototxt'
weights_path = '/home/ellerch/caffeProject/auto_trainer_output/2016-10-30_11h-46m-18s_experiment/vanilla_alexnet/_iter_110000.caffemodel'
out_path = '/home/ellerch/caffeProject/mean_visi.png'
mean_path = '/home/ellerch/db/places365/places365CNN_mean.binaryproto'

#convert to npy
blob = caffe.proto.caffe_pb2.BlobProto()
data = open( mean_path , 'rb' ).read()
blob.ParseFromString(data)
mean_arr = np.array( caffe.io.blobproto_to_array(blob) )

net = caffe.Classifier(model_path,
                       weights_path)
net.set_raw_scale('data', 255)
#channel_swap=(2, 1, 0),

net_mean = caffe.Classifier(model_path,
                            weights_path,
                            mean=mean_arr[0])
net_mean.set_raw_scale('data', 255)

#channel_swap=(2, 1, 0),
# mean=np.load('/path/to/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'),
fake = np.ones((227, 227, 3))

fake_pre = net_mean.preprocess('data', fake)
fake_re = net.deprocess('data', fake_pre)

mean_image = 1 - fake_re

plt.imshow(mean_image)
pylab.savefig(out_path)
