# https://gist.github.com/yassersouri/f617bf7eff9172290b4f
import caffe
import numpy as np
from scipy.misc import imshow

caffe.set_mode_cpu()

mean_path = '/home/ellerch/db/places365/places365CNN_mean.binaryproto'
#mean_path = '/path/to/mean.binaryproto'

#convert to npy
blob = caffe.proto.caffe_pb2.BlobProto()
data = open(mean_path, 'rb').read()
blob.ParseFromString(data)
mean_arr = np.array(caffe.io.blobproto_to_array(blob))
print mean_arr[0].shape
#>(3,256,256)
r_chan = mean_arr[0][0]
g_chan = mean_arr[0][1]
b_chan = mean_arr[0][2]

imshow(np.dstack((r_chan, g_chan, b_chan)))
imshow(np.dstack((g_chan, b_chan, r_chan)))
imshow(np.dstack((b_chan, r_chan, g_chan)))

