import sys
import getpass
import configargparse
import uuid
import time
import os
import json
from datetime import datetime, timedelta
import logging
import shutil
import requests
import subprocess


def get_networks_from_file(jobs_file_path):
    job_list = []
    with open(jobs_file_path, "r") as jobsfile:
        for filename in jobsfile:
            if filename.startswith('#'):
                continue
            job_list.append(filename.replace('\r\n', ''))
    #open txt with network names/paths
    return job_list


def get_args():
    configpath = os.path.join(os.path.dirname(__file__), 'config.ini')
    parser = configargparse.ArgParser(default_config_files=[configpath])
    parser.add_argument('-d', '--debug', help='Debug Mode', action='store_true')
    parser.add_argument('-j', '--jobs-file', type=str,
                        help='text file, each line should be one path to a solver file')
    parser.add_argument('-o', '--output-path', type=str,
                        help='Path, where auto_trainer will create an output-directory')
    #parser.set_defaults(DEBUG=True)

    args = parser.parse_args()

    return args


def draw_job_plot(caffe_log_path, log):
    if not os.path.exists(caffe_log_path):
        log.error('caffe logfile not found!')
        return
    log.info('plotting learning curve as png')
    plot_script = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'progress_plot.py')
    process = subprocess.Popen(['python', plot_script, caffe_log_path, os.path.dirname(caffe_log_path)],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    output = process.communicate()[0]
    returncode = process.returncode
    while returncode is None:
        time.sleep(2)
        returncode = process.returncode
    if returncode is not 0:
        log.error('plotter return code: '+str(process.returncode))
        log.error(output)
    else:
        log.info('plotter returned code {}, everything okay'.format(returncode))

