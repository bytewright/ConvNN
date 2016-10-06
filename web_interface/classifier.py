#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging
import numpy as np
import time
import caffe

log = logging.getLogger(__name__)


class NNClassifier:
    def __init__(self, gpu_mode, **kwargs):
        if gpu_mode:
            logging.info("using GPU mode")
            caffe.set_mode_gpu()
        else:
            logging.info("using CPU mode")
            caffe.set_mode_cpu()
        #self.load_classifier()
        log.info('starting classifier')
        self.net = None
        self.labels = None

    def set_neural_network(self, network_path, weight_path, db_mean_path, labels_path):
        log.info('loading db_mean_path:{}'.format(db_mean_path))
        blob = caffe.proto.caffe_pb2.BlobProto()
        data = open(db_mean_path, 'rb').read()
        blob.ParseFromString(data)
        arr = np.array(caffe.io.blobproto_to_array(blob))
        out = arr[0]
        log.debug(out.shape)
        log.info('loading network_path:{}'.format(network_path))
        log.info('loading weight_path:{}'.format(weight_path))
        self.net = caffe.Classifier(
            network_path,
            weight_path)
        #, mean=out
        #image_dims=(self.model_args['image_dim']), raw_scale=self.model_args['image_raw_scale'],
        #mean = np.load(db_mean_path).mean(1).mean(1), channel_swap = (2, 1, 0)
        log.info('loading labels_path:{}'.format(labels_path))
        labels = []
        with open(labels_path, 'r') as file:
            for line in file:
                label = (line.split(' ')[0])[3:]
                labels.append(label)
        log.info('loaded {} labels'.format(len(labels)))
        self.labels = labels

        return True

    def classify_image(self, image):
        try:
            starttime = time.time()
            log.info('classifing image...')
            scores = self.net.predict([image], oversample=True).flatten()
            endtime = time.time()
            logging.debug(scores)
            logging.debug(-scores)
            indices = (-scores).argsort()[:5]
            logging.debug(indices)
            predictions = []
            for index in indices:
                pred = '{}: {}'.format(index, self.labels[index])
                predictions.append(self.labels[index])
                logging.debug(pred)
            #predictions = self.labels[indices]
            logging.debug(predictions)
            meta = [
                (p, '%.5f' % scores[i])
                for i, p in zip(indices, predictions)
                ]
        except Exception as err:
            logging.info('Classification error: %s', err)
            return (False, 'Something went wrong when classifying the '
                           'image. Maybe try another one?')
        result = [True,
                  meta,
                  [(1, 'maxAccurate1'), (2, 'maxAccurate2')],
                  '%.3f' % (endtime - starttime)]
        return result
