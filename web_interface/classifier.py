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
    def __init__(self, gpu_mode, **kwargs):
        self.net = None
        self.labels = None
        self.name = ''
        self.type = ''
        if gpu_mode:
            self.my_log_info('starting classifier with GPU')
            caffe.set_mode_gpu()
        else:
            self.my_log_info('starting classifier with CPU')
            caffe.set_mode_cpu()
        
    def my_log_info(self, msg):
        log.info('{}:{}'.format(self.name, msg))

    def set_neural_network(self, params):
        self.name = params['name']
        self.type = params['type']
        self.my_log_info('loading db_mean_path:{}'.format(params['mean_db_path']))
        mean_npy = np.load(params['mean_db_path'])
        print mean_npy.shape

        #blob = caffe.proto.caffe_pb2.BlobProto()
        #data = open(params['mean_db_path'], 'rb').read()
        #blob.ParseFromString(data)
        #mean_arr = np.array(caffe.io.blobproto_to_array(blob))[0]
        self.my_log_info('loading network_path:{}'.format(params['network_path']))
        self.my_log_info('loading weight_path:{}'.format(params['weights_path']))
        #self.net = caffe.Classifier(params['network_path'], params['weights_path'],
        #                            image_dims=[256, 256], mean=mean_npy,
        #                            input_scale=1.0, raw_scale=255.0, channel_swap=[2, 1, 0])

        self.my_log_info('loading labels_path:{}'.format(params['class_labels']))
        labels = []
        with open(params['class_labels'], 'r') as file:
            for line in file:
                label = (line.split(';')[1])
                labels.append(label)
        self.my_log_info('loaded {} labels'.format(len(labels)))
        self.labels = labels

        return True

    def dummy_classify(self, image):
        if 'vgg' in self.name:
            starttime = time.time()
            time.sleep(5.48)
            self.my_log_info('classifing image...')
            endtime = time.time()
            result = [True,
                      [('/f/forest/broadleaf', 0.67110592),
                       ('/f/field/wild', 0.065907449),
                       ('/p/pasture', 0.059617899),
                       ('/f/forest_path', 0.046961967),
                       ('/r/rainforest', 0.046631806)],
                      '%.3f' % (endtime - starttime),
                      '{} ({})'.format(self.name, self.type)]
        else:
            starttime = time.time()
            time.sleep(3.24)
            self.my_log_info('classifing image...')
            endtime = time.time()
            result = [True,
                      [('white wolf, Arctic wolf, Canis lupus tundrarum', 0.35455278),
                       ('tusker', 0.18887606),
                       ('timber wolf, grey wolf, gray wolf, Canis lupus', 0.094796762),
                       ('Indian elephant, Elephas maximus', 0.089816146),
                       ('Irish wolfhound', 0.088667534)],
                      '%.3f' % (endtime - starttime),
                      '{} ({})'.format(self.name, self.type)]
        return result

    def classify_image(self, image):
        try:
            starttime = time.time()
            self.my_log_info('classifing image...')
            #scores = self.net.predict([image], oversample=True).flatten()
            scores = self.net.predict([image]).flatten()
            endtime = time.time()
            minutes, sec = divmod(endtime-starttime, 60)
            self.my_log_info('classification done in {}'.format('%02dm %02ds' % (minutes, sec)))
            indices = (-scores).argsort()[:5]

            predictions = []
            for index in indices:
                pred = '{}: {}'.format(index, self.labels[index])
                predictions.append(self.labels[index])

            #predictions = self.labels[indices]
            meta = [
                (p, '%.5f' % scores[i])
                for i, p in zip(indices, predictions)
                ]
            log.debug('classification result: {}'.format(meta))
        except Exception as err:
            logging.info('Classification error: %s', err)
            return (False, 'Something went wrong when classifying the '
                           'image. Maybe try another one?')
        result = [True,
                  meta,
                  [(1, 'maxAccurate1'), (2, 'maxAccurate2')],
                  '%.3f' % (endtime - starttime)]
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
