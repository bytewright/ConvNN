import configargparse
import time
import os
import subprocess

CAFFE_TOOL_PATH = '/home/ellerch/bin/caffe/python/'
MY_TOOLS_PATH = '/home/ellerch/caffeProject/auto_trainer/caffe_tools/'

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
        log.info('plotter exited successfully (code {})'.format(returncode))
    return output


def draw_job_net(solver_path, output_file, log):
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
        log.error('draw_net return code: ' + str(process.returncode))
        log.error(output)
    else:
        log.info('draw_net exited successfully (code {})'.format(returncode))
    return output


def generate_parsed_splitted_logs(caffe_log_file, job_output_dir, log):
    #./parse_log.sh <input_log> <output_path>
    script_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'caffe_tools/', 'parse_log.sh')
    log.debug('calling {}'.format(script_path))
    process = subprocess.Popen([script_path,
                                caffe_log_file,
                                job_output_dir],
                               shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    output = process.communicate()[0]
    returncode = process.returncode
    while returncode is None:
        time.sleep(2)
        returncode = process.returncode
    if returncode is not 0:
        log.error('parse_log return code: ' + str(process.returncode))
        log.error(output)
    else:
        log.info('parse_log exited successfully (code {})'.format(returncode))
    return output
