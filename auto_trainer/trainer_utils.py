import shutil
import sys
import configargparse
import time
import os
import subprocess
import logging
import caffe
from PIL import Image
import numpy as np
from scipy.misc import toimage

#CAFFE_TOOL_PATH = '/home/ellerch/bin/caffe/python/'
#MY_TOOLS_PATH = '/home/ellerch/caffeProject/auto_trainer/caffe_tools/'


def get_args():
    configpath = os.path.join(os.path.dirname(__file__), 'config.ini')
    parser = configargparse.ArgParser(default_config_files=[configpath])
    parser.add_argument('-d', '--debug', help='Debug Mode', action='store_true')
    parser.add_argument('-j', '--jobs-file', type=str,
                        help='text file, each line should be one path to a solver file')
    parser.add_argument('-j', '--caffe-path', type=str,
                        help='text file, each line should be one path to a solver file')
    parser.add_argument('-o', '--output-path', type=str,
                        help='Path, where auto_trainer will create an output-directory')
    #parser.set_defaults(DEBUG=True)

    args = parser.parse_args()

    return args


def get_networks_from_file(jobs_file_path):
    job_list = []
    # open txt with network names/paths
    with open(jobs_file_path, "r") as jobsfile:
        for job_desc in jobsfile:
            job_desc = job_desc.rstrip()  # remove '\n' at end of line
            if job_desc.startswith('#'):
                continue
            if not os.path.isfile(os.path.join(job_desc, 'solver.prototxt')):
                logging.error('no solver.prototxt in \n"{}",\nskipping job'.format(job_desc))
                continue
            job_list.append(job_desc)
    return job_list


def check_job(job):
    if job['ignore']:
        logging.info('ignoring job: ' + job['name'])
        return None
    if 'solver_path' not in job:
        logging.error('no valid solver_path key in job "{}",\nskipping job'.format(job))
        return None
    if not os.path.isfile(job['solver_path']):
        logging.error('no solver at \n"{}",\nskipping job'.format(job['solver_path']))
        return None
    net_path = ''
    snapshot_path = ''
    with open(job['solver_path'], 'r') as search:
        for line in search:
            line = line.rstrip()  # remove '\n' at end of line
            if line.startswith('snapshot_prefix: '):
                snapshot_path = line.replace('snapshot_prefix: ', '').replace('"', '')
            if line.startswith('net: '):
                net_path = line.replace('net: ', '').replace('"', '')
    if net_path is not '' and snapshot_path is not '':
        job['model_path'] = net_path
        job['snapshot_path'] = snapshot_path
    else:
        logging.error('no valid model_path or snapshot_path in {}-solver, skipping job'.format(job['name']))
        return None
    if not os.path.isfile(job['model_path']):
        logging.error('model_path in {}-solver is no file, skipping job'.format(job['name']))
        return None
    return job


def extract_filters(network_path, weight_path, output_path):
    if not os.path.exists(weight_path):
        logging.error('weights not found!: ' + weight_path)
        return
    logging.info('extracting filters')
    filter_extract_script = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                         'my_tools',
                                         'filter_extractor.py')
    process = subprocess.Popen(['python', filter_extract_script,
                                network_path, weight_path, output_path],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    output = process.communicate()[0]
    returncode = process.returncode
    while returncode is None:
        time.sleep(2)
        returncode = process.returncode
    if returncode is not 0:
        logging.error('extractor return code: '+str(process.returncode))
        logging.error(output)
    else:
        logging.info('extractor exited successfully (code {})'.format(returncode))


def draw_job_plot(caffe_log_path, output_file):
    if not os.path.exists(caffe_log_path):
        logging.error('caffe logfile not found!')
        return
    logging.info('plotting learning curve as png')
    plot_script = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'progress_plot.py')
    process = subprocess.Popen(['python', plot_script, caffe_log_path, output_file],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    output = process.communicate()[0]
    #log.debug(output)
    returncode = process.returncode
    while returncode is None:
        time.sleep(2)
        returncode = process.returncode
    if returncode is not 0:
        logging.error('plotter return code: '+str(process.returncode))
        logging.error(output)
    else:
        logging.info('plotter exited successfully (code {})'.format(returncode))


def draw_job_plot2(test_log_path, output_file):
    if not os.path.exists(test_log_path):
        logging.error('test logfile not found!')
        return
    logging.info('plotting learning curve as png')
    plot_script = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'my_tools', 'progress_plot.py')
    process = subprocess.Popen(['python',
                                plot_script,
                                '--plot_data',
                                test_log_path,
                                '--output_png_path',
                                output_file],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    output = process.communicate()[0]
    returncode = process.returncode
    while returncode is None:
        time.sleep(2)
        returncode = process.returncode
    if returncode is not 0:
        logging.error('plotter return code: '+str(process.returncode))
        logging.error(output)
    else:
        logging.info('plotter exited successfully (code {})'.format(returncode))
    return output


def draw_job_net(solver_path, output_file, CAFFE_TOOL_PATH):
    # draw_net.py <netprototxt_filename> <out_img_filename>
    net_path = ''
    with open(solver_path, 'r') as search:
        for line in search:
            line = line.rstrip()  # remove '\n' at end of line
            if line.startswith('net: '):
                net_path = line.replace('net: ', '').replace('"', '')
    process = subprocess.Popen(['python',
                                CAFFE_TOOL_PATH + 'draw_net.py',
                                net_path,
                                output_file],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    output = process.communicate()[0]
    returncode = process.returncode
    while returncode is None:
        time.sleep(2)
        returncode = process.returncode
    if returncode is not 0:
        logging.error('draw_net return code: ' + str(process.returncode))
        logging.error(output)
    else:
        logging.info('draw_net exited successfully (code {})'.format(returncode))
    return output


def generate_parsed_splitted_logs(caffe_log_file, job_output_dir):
    #./parse_log.sh <input_log> <output_path>
    script_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'caffe_tools', 'parse_log.sh')
    process = subprocess.Popen(['{} {} {}'.format(script_path, caffe_log_file, job_output_dir)],
                               shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    output = process.communicate()[0]
    returncode = process.returncode
    while returncode is None:
        time.sleep(2)
        returncode = process.returncode
    if returncode is not 0:
        logging.error('parse_log return code: ' + str(process.returncode))
        logging.error(output.replace(': ', ':\n'))
    else:
        logging.info('parse_log exited successfully (code {})'.format(returncode, output))
    return output


def generate_output_directory(solver_path, net_path, snapshot_path):
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
        # copy used settings and network to output dir
        shutil.copyfile(solver_path, os.path.join(snapshot_path, os.path.basename(solver_path)))
        shutil.copyfile(net_path, os.path.join(snapshot_path, os.path.basename(net_path)))
    else:
        logging.error('tmp directory is not empty! Aborting')
        sys.exit()
    return snapshot_path
