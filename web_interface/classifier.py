#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging
import numpy as np
import cStringIO as StringIO
import time
import caffe
import urllib
import os
from threading import Thread

log = logging.getLogger(__name__)


class NNClassifier(Thread):
    def __init__(self, params, gpu_mode, **kwargs):
        Thread.__init__(self)

        self.name = params['name']
        self.type = params['type']
        self.path_dict = {}
        param_fields = ['network_path', 'weights_path', 'mean_db_path', 'class_labels']
        for param_name in param_fields:
            if os.path.isfile(params[param_name]) and os.path.exists(params[param_name]):
                self.path_dict[param_name] = params[param_name]
            else:
                self.my_log_error('No File found at:\n{}'.format(params[param_name]))
        self.gpu_mode = gpu_mode
        self.image = None
        self.image_is_set = False
        self.result = None

    def set_image(self, image):
        logging.debug(type(image))
        self.image = image
        self.image_is_set = True
        return True

    def set_image_url(self, img_url):
        try:
            string_buffer = StringIO.StringIO(
                urllib.urlopen(img_url).read())
            self.image = caffe.io.load_image(string_buffer)

        except Exception as err:
            # For any exception we encounter in reading the image, we will just
            # not continue.
            logging.info('URL Image open error: %s', err)
            return False
        self.image_is_set = True
        return True

    def run(self):
        if not self.image_is_set:
            self.my_log_error('Image has not been set, aborting classification!')
            return
        # init network
        if self.gpu_mode:
            self.my_log_info('starting classifier with GPU')
            caffe.set_mode_gpu()
        else:
            self.my_log_info('starting classifier with CPU')
            caffe.set_mode_cpu()

        # load files
        mean_npy = np.load(self.path_dict['mean_db_path'])

        # str(path) is needed to match c++ interface....
        net = caffe.Classifier(str(self.path_dict['network_path']), str(self.path_dict['weights_path']),
                               image_dims=[256, 256], mean=mean_npy,
                               input_scale=1.0, raw_scale=255.0, channel_swap=[2, 1, 0])

        labels = []
        with open(self.path_dict['class_labels'], 'r') as labels_file:
            for line in labels_file:
                if line.split(';').__len__() is not 2:
                    self.my_log_error('labels file not corectly formated. Expected: <id>;<label>\n'
                                      'line read:{}'.format(line))
                label = (line.split(';')[1]).rstrip()
                labels.append(label)
        self.my_log_info('loaded {} labels'.format(len(labels)))
        labels = labels
        # start classifing
        try:
            starttime = time.time()
            self.my_log_info('classifing image...')
            scores = net.predict([self.image], oversample=True).flatten()
            #scores = self.net.predict([image]).flatten()
            endtime = time.time()
            minutes, sec = divmod(endtime-starttime, 60)
            self.my_log_info('classification done in {}'.format('%02dm %02ds' % (minutes, sec)))
            indices = (-scores).argsort()[:5]

            predictions = []
            for index in indices:
                predictions.append(labels[index])

            meta = [
                (p, '%.5f' % scores[i])
                for i, p in zip(indices, predictions)
                ]
        except Exception as err:
            logging.info('Classification error: %s', err)
            self.result = (False, 'Something went wrong when classifying the '
                           'image. Maybe try another one?')
            return
        self.result = [True,
                       meta,
                       '%.3f' % (endtime - starttime),
                       '{} ({})'.format(self.name, self.type)]

    def get_result(self):
        if self.result is None:
            return False, None
        return True, self.result

    def get_name(self):
        return self.name

    def my_log_info(self, msg):
        log.info('{}:{}'.format(self.name, msg))

    def my_log_error(self, msg):
        log.error('{}:{}'.format(self.name, msg))
