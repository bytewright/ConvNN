#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging
import numpy as np
import cStringIO as StringIO
import time
import caffe
import urllib

log = logging.getLogger(__name__)


class NNClassifier:
    def __init__(self, **kwargs):
        self.net = None
        self.labels = None
        self.name = ''
        self.type = ''

    def my_log_info(self, msg):
        log.info('{}:{}'.format(self.name, msg))

    def set_neural_network(self, params, gpu_mode):
        self.name = params['name']
        self.type = params['type']

        if gpu_mode:
            self.my_log_info('starting classifier with GPU')
            caffe.set_mode_gpu()
        else:
            self.my_log_info('starting classifier with CPU')
            caffe.set_mode_cpu()

        self.my_log_info('loading db_mean_path:\n{}'.format(params['mean_db_path']))
        mean_npy = np.load(params['mean_db_path'])

        self.my_log_info('loading network_path:\n{}'.format(params['network_path']))
        self.my_log_info('loading weight_path:\n{}'.format(params['weights_path']))
        # str(path) is needed to match c++ interface....
        self.net = caffe.Classifier(str(params['network_path']), str(params['weights_path']),
                                    image_dims=[256, 256], mean=mean_npy,
                                    input_scale=1.0, raw_scale=255.0, channel_swap=[2, 1, 0])

        self.my_log_info('loading labels_path:\n{}'.format(params['class_labels']))
        labels = []
        with open(params['class_labels'], 'r') as file:
            for line in file:
                label = (line.split(';')[1]).rstrip()
                labels.append(label)
        self.my_log_info('loaded {} labels'.format(len(labels)))
        self.labels = labels
        return True

    def classify_image(self, image):
        try:
            starttime = time.time()
            self.my_log_info('classifing image...')
            scores = self.net.predict([image], oversample=True).flatten()
            #scores = self.net.predict([image]).flatten()
            endtime = time.time()
            minutes, sec = divmod(endtime-starttime, 60)
            self.my_log_info('classification done in {}'.format('%02dm %02ds' % (minutes, sec)))
            indices = (-scores).argsort()[:5]

            predictions = []
            for index in indices:
                #pred = '{}: {}'.format(index, self.labels[index])
                predictions.append(self.labels[index])

            #predictions = self.labels[indices]
            meta = [
                (p, '%.5f' % scores[i])
                for i, p in zip(indices, predictions)
                ]
            #log.debug('classification result: {}'.format(meta))
        except Exception as err:
            logging.info('Classification error: %s', err)
            return (False, 'Something went wrong when classifying the '
                           'image. Maybe try another one?')
        result = [True,
                  meta,
                  '%.3f' % (endtime - starttime),
                  '{} ({})'.format(self.name, self.type)]
        return result


    def classify_url(self, img_url):
        try:
            string_buffer = StringIO.StringIO(
                urllib.urlopen(img_url).read())
            image = caffe.io.load_image(string_buffer)

        except Exception as err:
            # For any exception we encounter in reading the image, we will just
            # not continue.
            logging.info('URL Image open error: %s', err)
            return (False, 'Something went wrong when classifying the '
                           'image. Maybe try another one?')

        return self.classify_image(image)

