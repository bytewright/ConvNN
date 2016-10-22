import caffe
import os
from caffe import layers


def create_neural_net(batch_size=50):
    net = caffe.NetSpec()
    net.data, net.label = layers.HDF5Data(batch_size=batch_size, source="blabla", ntop=2)

    net.fc1 = layers.InnerProduct(net.data, num_output=100, weight_filler=dict(type='xavier'))
    net.relu1 = layers.ReLU(net.fc1, in_place=True)
    net.fc2 = layers.InnerProduct(net.relu1, num_output=50, weight_filler=dict(type='xavier'))
    net.relu2 = layers.ReLU(net.fc2, in_place=True)
    net.fc3 = layers.InnerProduct(net.relu1, num_output=20, weight_filler=dict(type='xavier'))
    net.relu3 = layers.ReLU(net.fc3, in_place=True)
    net.fc4 = layers.InnerProduct(net.relu3, num_output=1, weight_filler=dict(type='xavier'))
    net.loss = layers.SoftmaxWithLoss(net.fc4, net.label)
    return 'testnet', net.to_proto()

if __name__=='__main__':

    batch_size = 50
    output_name, prototxt = create_neural_net()
    out_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), output_name)
    print 'writing net {} to\n{}'.format(output_name, out_path)
    with open(out_path, 'w') as f:
        f.write(str(prototxt))
