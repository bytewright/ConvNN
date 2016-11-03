#!/usr/bin/python
# -*- coding: utf-8 -*-

import datetime
import logging
import os
import json
import configargparse
from server import NNWebInterface
from classifier import NNClassifier

logFormatter = logging.Formatter("%(asctime)s [%(module)14s] [%(levelname)5s] %(message)s")
log = logging.getLogger()
log.setLevel(logging.DEBUG)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
log.addHandler(consoleHandler)


def get_args():
    configpath = os.path.join(os.path.dirname(__file__), 'config.ini')
    parser = configargparse.ArgParser(default_config_files=[configpath])
    parser.add_argument('-d', '--debug', help='Debug Mode', action='store_true')
    parser.add_argument('-np', '--cnns_json', type=str, help='Debug Mode')
    parser.add_argument('-p', '--port', type=int, help='Debug Mode')
    parser.add_argument('-ae', '--allowed_extensions',
                        help='Path, where auto_trainer will create an output-directory')
    parser.add_argument('-up', '--upload_path', type=str,
                        help='Path, where auto_trainer will create an output-directory')
    return parser.parse_args()


def run_self_test():
    log.info('allowed_extensions: {}'.format(set(args.allowed_extensions.split(','))))
    log.info('upload_path: {}'.format(args.upload_path))

if __name__ == '__main__':
    args = get_args()
    if args.debug:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)
    file_name = 'app_log(' + datetime.datetime.now().strftime('%Y-%m-%d_%Hh-%Mm-%Ss') + ').log'
    fileHandler = logging.FileHandler(os.path.join(os.path.dirname(__file__), 'logs', file_name))
    fileHandler.setFormatter(logFormatter)
    log.addHandler(fileHandler)
    run_self_test()

    # load cnns
    classifiers = []
    cnns_list = json.load(open(args.cnns_json, 'r'))
    for cnn_index in cnns_list:
        # for each cnn, crate classifier
        classifier = NNClassifier(gpu_mode=False)
        if not classifier.set_neural_network(cnns_list[cnn_index]):
            log.error('classifier {} could not be loaded'.format(cnns_list[cnn_index]['name']))
        else:
            classifiers.append(classifier)
    if classifiers.__len__() <= 0:
        log.error('no valid classifier in '+args.cnns_json)
    app = NNWebInterface('webinterface', classifiers, args.upload_path)
    app.run(host='0.0.0.0', port=args.port)
