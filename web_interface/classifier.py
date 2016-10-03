#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging

log = logging.getLogger(__name__)

class NNClassifier():
    def __init__(self, **kwargs):
        log.info('starting classifier')

    def set_neural_network(self, network_path, weight_path, db_mean_path):
        log.info('loading network_path:{}'.format(network_path))
        log.info('loading weight_path:{}'.format(weight_path))
        log.info('loading db_mean_path:{}'.format(db_mean_path))

        return True

    def classify_image(self, image):
        result = [True, 'maxSpecific', 'maxAccurate', 'duration']
        return result
