import json
import logging
import os
import shutil
import subprocess
import time


#def get_networks_from_file(jobs_file_path):
#    job_list = []
#    # open txt with network names/paths
#    with open(jobs_file_path, "r") as jobsfile:
#        for job_desc in jobsfile:
#            job_desc = job_desc.rstrip()  # remove '\n' at end of line
#            if job_desc.startswith('#'):
#                continue
#            if not os.path.isfile(os.path.join(job_desc, 'solver.prototxt')):
#                logging.error('no solver.prototxt in \n"{}",\nskipping job'.format(job_desc))
#                continue
#            job_list.append(job_desc)
#    return job_list


def check_job(job):
    # checks all paths in solver file
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
    # parse network and snapshot path from solver
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
    # needs adjustment for new projects
    filter_extract_script = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                         'tools',
                                         'filter_extractor.py')
    if not os.path.isfile(filter_extract_script):
        logging.error('filter_extract_script not found at {}'.format(filter_extract_script))
        return
    if not os.path.exists(weight_path):
        logging.error('weights not found!: ' + weight_path)
        return
    logging.info('extracting filters')
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


#def draw_job_plot(caffe_log_path, output_file):
#    # needs adjustment for new projects
#    plot_script = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'progress_plot.py')
#
#    if not os.path.exists(caffe_log_path):
#        logging.error('caffe logfile not found!')
#        return
#    logging.info('plotting learning curve as png')
#    process = subprocess.Popen(['python', plot_script, caffe_log_path, output_file],
#                               stdout=subprocess.PIPE,
#                               stderr=subprocess.STDOUT)
#    output = process.communicate()[0]
#    #log.debug(output)
#    returncode = process.returncode
#    while returncode is None:
#        time.sleep(2)
#        returncode = process.returncode
#    if returncode is not 0:
#        logging.error('plotter return code: '+str(process.returncode))
#        logging.error(output)
#    else:
#        logging.info('plotter exited successfully (code {})'.format(returncode))
#
#
#def draw_job_plot2(test_log_path, output_file):
#    if not os.path.exists(test_log_path):
#        logging.error('test logfile not found!')
#        return
#    logging.info('plotting learning curve as png')
#    plot_script = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tools', 'progress_plot.py')
#    process = subprocess.Popen(['python',
#                                plot_script,
#                                '--plot_data',
#                                test_log_path,
#                                '--output_png_path',
#                                output_file],
#                               stdout=subprocess.PIPE,
#                               stderr=subprocess.STDOUT)
#    output = process.communicate()[0]
#    returncode = process.returncode
#    while returncode is None:
#        time.sleep(2)
#        returncode = process.returncode
#    if returncode is not 0:
#        logging.error('plotter return code: '+str(process.returncode))
#        logging.error(output)
#    else:
#        logging.info('plotter exited successfully (code {})'.format(returncode))
#    return output
#

def draw_job_plot(test_log_path, output_file):
    # needs adjustment for new projects
    plot_script = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tools', 'progress_plot2.py')
    if not os.path.isfile(plot_script):
        logging.error('plotting script not found at {}'.format(plot_script))
        return
    if not os.path.exists(test_log_path):
        logging.error('test logfile not found!')
        return
    logging.info('plotting learning curve as png')
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


def draw_job_net(solver_path, output_file, caffe_python_path):
    # needs adjustment for new projects
    # caffe_python_path is defined in config.ini
    net_draw_script = os.path.join(caffe_python_path, 'draw_net.py')
    if not os.path.isfile(net_draw_script):
        logging.error('net_draw_script not found at {}'.format(net_draw_script))
        return
    # draw_net.py <netprototxt_filename> <out_img_filename>
    net_path = ''
    with open(solver_path, 'r') as search:
        for line in search:
            line = line.rstrip()  # remove '\n' at end of line
            if line.startswith('net: '):
                net_path = line.replace('net: ', '').replace('"', '')
    process = subprocess.Popen(['python',
                                net_draw_script,
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
    if not os.path.isfile(script_path):
        logging.error('script_path not found at {}'.format(script_path))
        return
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


def generate_parsed_splitted_logs2(caffe_log_file, job_output_dir):
    script_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tools', 'csv_log_parser.py')
    #script_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'scripts',
    #                           'log_parser', 'my_log_parser2.py')
    if not os.path.isfile(script_path):
        logging.error('script_path not found at {}'.format(script_path))
        return
    process = subprocess.Popen(['python',
                                script_path,
                                '--log_file',
                                caffe_log_file,
                                '--output_file',
                                job_output_dir],
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
        logging.error('{} already exists in tmp directory ! Aborting')
        return None, False
    return snapshot_path, True


def get_training_stats(csv_path):
    # needs adjustments if another csv parser is used
    csv_delimiter = ';'
    acc1_index = 1
    acc5_index = 2
    loss_index = 3
    num_items = 10

    test_data = []
    with open(csv_path, 'r') as f:
        for line in f.readlines():
            # print line
            line = line.rstrip()
            test_data.append([data for data in line.split(csv_delimiter) if data is not ''])

    csv_data = test_data[1:]

    if csv_data.__len__() < num_items:
        num_items = int(csv_data.__len__() / 2)

    acc1 = [float(x[acc1_index]) for x in csv_data]
    acc5 = [float(x[acc5_index]) for x in csv_data]
    loss = [float(x[loss_index]) for x in csv_data]
    max_top1 = (max(acc1))
    max_top5 = (max(acc5))
    avg_loss = 0.0
    avg_acc_top1 = 0.0
    avg_acc_top5 = 0.0

    for i in range(num_items+1):
        avg_acc_top1 += acc1[-(i+1)] / num_items
        avg_acc_top5 += acc5[-(i+1)] / num_items
        avg_loss += loss[-(i+1)] / num_items

    stats = {'max_top1': round(max_top1 * 100, 3),
             'max_top5': round(max_top5 * 100, 3),
             'avg_loss': round(avg_loss, 3),
             'avg_acc_top1': round(avg_acc_top1 * 100, 3),
             'avg_acc_top5': round(avg_acc_top5 * 100, 3)}
    return stats


#def get_avg_acc_and_loss(log_path):
#    delimiter = ' '
#    test_data = []
#    with open(log_path, 'r') as dest_f:
#        for line in dest_f.readlines():
#            line = line.rstrip()
#            test_data.append([data for data in line.split(delimiter) if data is not ''])
#    if test_data.__len__() < 10:
#        logging.info('found only {} lines in log'.format(test_data.__len__()))
#        use_last_n = 1.0
#    else:
#        use_last_n = 10.0
#    avg_acc = 0.0
#    avg_loss = 0.0
#    found_data_points = 0
#    for data in reversed(test_data):
#        if found_data_points == use_last_n:
#            break
#        if len(data) < 4:
#            continue
#        avg_acc += float(data[2]) / use_last_n
#        avg_loss += float(data[3]) / use_last_n
#        found_data_points += 1
#    return avg_acc, avg_loss


def get_best_caffemodel(snapshot_path):
    best_file = 1
    best_file_name = ''
    for f in os.listdir(snapshot_path):
        if f.endswith(".caffemodel"):
            iter_num = int(f.replace('_iter_', '').replace('.caffemodel', ''))
            if iter_num > best_file:
                best_file = iter_num
                best_file_name = f
    path_to_file = os.path.join(snapshot_path, best_file_name)
    if os.path.isfile(path_to_file):
        return path_to_file
    return None


def save_job_stats_to_json(job):
    #logging.info('Job "{}" completed in {}. Accuracy: {:.3f}'.format(job['name'],
    #                                                             job['duration'],
    #                                                             job['accuracy']))
    log_path = os.path.join(job['snapshot_path'], job['name'].replace(' ', '_') + "_stats.json")
    json.dump(job, open(log_path, 'w'), sort_keys=True, indent=4, separators=(',', ': '))
    return


def move_all_files_from_to(src_path, dest_path):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    else:
        logging.error('there already is a completed job with name {} in output dir!'.format(dest_path))
        return False
    file_list = os.listdir(src_path)
    logging.info('moving all files from\n{}\nto\n{}'.format(src_path, dest_path))
    for i in file_list:
        src = os.path.join(src_path, i)
        dest = os.path.join(dest_path, i)
        if os.path.exists(dest):
            if os.path.isdir(dest):
                # clean out subfolders
                move_all_files_from_to(src, dest)
                continue
            else:
                os.remove(dest)
        shutil.move(src, dest_path)
    os.rmdir(src_path)
    return True