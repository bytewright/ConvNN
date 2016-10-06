#!/usr/bin/python
# -*- coding: utf-8 -*-

import datetime
import logging
import os
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
    parser.add_argument('-np', '--network_path', type=str, help='Debug Mode')
    parser.add_argument('-nw', '--weights_path', type=str, help='Debug Mode')
    parser.add_argument('-l', '--class_labels', type=str,
                        help='text file, each line should be one path to a solver file')
    parser.add_argument('-m', '--db_mean', type=str,
                        help='Path, where auto_trainer will create an output-directory')
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
    classifier = NNClassifier(gpu_mode=False)
    if not classifier.set_neural_network(args.network_path,
                                         args.weights_path,
                                         args.db_mean,
                                         args.class_labels):
        log.error('classifier could not be loaded')
    else:
        app = NNWebInterface('webinterface', classifier, args.upload_path)
        other_classifiers = [(args.network_path, args.weights_path, args.db_mean),
                       (args.network_path, args.weights_path, args.db_mean)]
        app.set_other_classifiers(other_classifiers)
        app.run()
