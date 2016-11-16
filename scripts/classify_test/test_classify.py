import json
import logging
import caffe
from classifier import NNClassifier

logFormatter = logging.Formatter("%(asctime)s [%(module)14s] [%(levelname)5s] %(message)s")
log = logging.getLogger()
log.setLevel(logging.DEBUG)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
log.addHandler(consoleHandler)
json_path = '/home/ellerch/caffeProject/web_interface/cnn/cnns.json'
img_path = '/home/ellerch/caffeProject/scripts/classify_test/pzjxnad5.png'

if __name__ == '__main__':
    classifiers = []
    inputs = caffe.io.load_image(img_path)
    cnns_list = json.load(open(json_path, 'r'))
    for cnn_index in cnns_list:
        # for each cnn, crate classifier
        classifier = NNClassifier()
        if not classifier.set_neural_network(cnns_list[cnn_index], gpu_mode=False):
            log.error('classifier {} could not be loaded'.format(cnns_list[cnn_index]['name']))
        else:
            classifiers.append(classifier)
    for classi in classifiers:
        for result in classi.classify_image(inputs):
            print result
