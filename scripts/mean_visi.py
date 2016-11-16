# https://gist.github.com/yassersouri/f617bf7eff9172290b4f
import caffe
import numpy as np
import random
import os
from scipy import misc

def cut_nxn(img, n):
    #print img.shape
    w,h,c = img.shape
    if not c is 3:
        print img.shape
    cut_img=img[w/2-n/2:w/2+n/2,h/2-n/2:h/2+n/2]
    return cut_img

caffe.set_mode_cpu()

mean_path = '/home/ellerch/db/places365/places365CNN_mean.binaryproto'
#mean_path = '/path/to/mean.binaryproto'
if __name__ == "__main__":
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
    print '(min,max):\n({},{})\n({},{})\n({},{})'.format(np.min(r_chan), np.max(r_chan),
                                                      np.min(g_chan), np.max(g_chan),
                                                      np.min(b_chan), np.max(b_chan))
    mean_png = np.dstack((b_chan, g_chan, r_chan))
    misc.imshow(mean_png)

if False:
    img_dir = '/home/ellerch/db/places365/test_large'
    img_count = 0
    for file in os.listdir(img_dir):
        if file.endswith('jpg'):
            img_count+=1
    print 'got {} imgs'.format(img_count)
    comp_img = None
    comp_count = 1000
    used_count = 0
    for i in range(comp_count):
        if i % 100 is 0:
            print 'iter {}'.format(i)
        rnd = random.randint(0,img_count)
        img_path = os.path.join(img_dir, os.listdir(img_dir)[rnd])
        img = misc.imread(img_path)
        if img.shape.__len__()<3:
            print 'sw img'
            continue
        else:
            used_count += 1
        img = cut_nxn(img, 256)
        if comp_img is None:
            comp_img = img
        else:
            comp_img += img
    misc.imshow(comp_img/used_count)


