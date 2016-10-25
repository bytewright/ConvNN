import caffe
import os
from caffe import layers
from caffe import params

def create_neural_net(batch_size=50):
    net_name = 'Alexnet'
    net = caffe.NetSpec()
    net.data, net.label = layers.HDF5Data(batch_size=batch_size, source="blabla", ntop=2)
    net.conv1 = layers.Convolution(net.data, kernel_size=, num_output=, stride=, pad=, weight_filler=dict(type='xavier'))
    net.conv1 = layers.Convolution(bottom, param = [dict(lr_mult=), dict(lr_mult=)],
                                   kernel_h = , kernel_w = , stride = , num_output = , pad = ,
                                   weight_filler = dict(type='gaussian', std=0.1, sparse=sparse),
                                   bias_filler = dict(type='constant', value=0))
    net.relu1 = layers.ReLU(net.conv1, in_place=True)
    net.pool1 = layers.Pooling(net.relu1, kernel_size=2, stride=2, pool=params.Pooling.MAX)
    net.lrn1
    net.conv2
    net.relu2 = layers.ReLU(net.conv2, in_place=True)
    net.pool2 = layers.Pooling(net.relu2, kernel_size=2, stride=2, pool=params.Pooling.MAX)
    net.lrn2
    net.conv3
    net.relu3 = layers.ReLU(net.conv3, in_place=True)
    net.conv4
    net.relu4 = layers.ReLU(net.fc1, in_place=True)
    net.conv5
    net.pool3 = layers.Pooling(net.conv5, kernel_size=2, stride=2, pool=params.Pooling.MAX)
    net.relu5 = layers.ReLU(net.fc1, in_place=True)
    net.fc1 = layers.InnerProduct(net.data, num_output=100, weight_filler=dict(type='xavier'))
    net.relu6 = layers.ReLU(net.fc1, in_place=True)
    net.dropout1
    net.fc2 = layers.InnerProduct(net.relu1, num_output=50, weight_filler=dict(type='xavier'))
    net.relu2 = layers.ReLU(net.fc2, in_place=True)
    net.dropout2
    net.fc3 = layers.InnerProduct(net.relu1, num_output=20, weight_filler=dict(type='xavier'))
    net.loss = layers.SoftmaxWithLoss(net.fc4, net.label)
    net.accuracy
    return net_name+'.prototxt', net.to_proto()

if __name__=='__main__':

    batch_size = 50
    output_name, prototxt = create_neural_net()
    out_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), output_name)
    print 'writing net {} to\n{}'.format(output_name, out_path)
    with open(out_path, 'w') as f:
        f.write(str(prototxt))
