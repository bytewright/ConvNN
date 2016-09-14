import logging
import json
import subprocess
import time
import os
import datetime
import sys
import shutil
from NNTrainClsSub import NetworkTrainer
from trainer_utils import get_networks_from_file, get_args, draw_job_plot
from net_creator import example_lenet

logFormatter = logging.Formatter("%(asctime)s [%(module)14s] [%(levelname)5s] %(message)s")
log = logging.getLogger()
#logging.basicConfig(format='%(asctime)s [%(module)14s] [%(levelname)5s] %(message)s')
#log = logging.getLogger()
log.setLevel(logging.DEBUG)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
log.addHandler(consoleHandler)

caffePythonToolsPath = '/home/ellerch/bin/caffe/python/'


def generate_job_log(output_dir, job_index, stats_dict):
    log.info('Job "{}" completed in {:.2f}s. Accuracy: {:.3f}'.format(stats_dict['name'],
                                                                      stats_dict['duration'],
                                                                      stats_dict['accuracy']))
    log_path = os.path.join(output_dir, "stats.json".format(job_index))
    json.dump(stats_dict, open(log_path, 'w'))
    return


def train_network(thread_name, job_file):
    log.debug('Name: {}, my job file: {}'.format(thread_name, job_file))
    return


def get_networks_from_python():
    #with open('examples/mnist/lenet_auto_train.prototxt', 'w') as f:
    #    f.write(str(example_lenet('examples/mnist/mnist_train_lmdb', 64)))

    #with open('examples/mnist/lenet_auto_test.prototxt', 'w') as f:
    #    f.write(str(example_lenet('examples/mnist/mnist_test_lmdb', 100)))
    # todo make solver file
    return []


def draw_net_for_job(net_prototxt_path, output_path):
    # draw_net.py <netprototxt_filename> <out_img_filename>
    draw_net_process = subprocess.Popen(['python',
                                         caffePythonToolsPath + 'draw_net.py',
                                         net_prototxt_path,
                                         output_path + 'net.png'],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT)
    output = draw_net_process.communicate()[0]
    return output


def train_networks(network_list, output_path):
    train_threads = []
    for job in network_list:
        log.debug('reading job desc: ' + job)
        if job.startswith('#'):
            log.info('skipping comment: ' + job)
            continue

        solverProto = job + 'solver.prototxt'
        if not open(solverProto, 'r'):
            log.error('can\'t open solver.prototxt!, skipping job')
            continue

        jobID = network_list.index(job)
        job_output_dir = output_path + '/job{}/'.format(jobID)
        os.makedirs(job_output_dir)

        log.info('starting thread for job {}'.format(jobID))
        train_thread = NetworkTrainer(job, job_output_dir, solverProto, log)
        train_threads.append(train_thread)
        train_thread.daemon = True
        train_thread.setName('thread {}'.format(jobID))
        train_thread.start()

        # generate image of NN
        time.sleep(2)  # wait for trainer to get started
        draw_net_for_job(job + 'net_train.prototxt', job_output_dir)

        train_thread.join()
        # traning is done, write log and other output
        generate_job_log(job_output_dir, jobID, train_thread.get_stats())
        draw_job_plot(os.path.join(job_output_dir, "caffe_training.log"), log)
        # time.sleep(2)
        # log.debug("testing classify")
        # proc = subprocess.Popen(['python', caffePythonToolsPath + 'classify.py',
        #                         'input_file', validationFolder,
        #                         ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        # output = proc.communicate()[0]

    log.info('all jobs completed')
    for job in train_threads:
        log.info('thread {}: completed in {}s'.format(job.getName(), job.get_duration()))


def generate_output_directories(network_list):
    generated_dirs = []
    for network in network_list:
        solver_path = os.path.join(network, 'solver.prototxt')
        if not os.path.isfile(solver_path):
            log.error(solver_path + ' does not exist!, skipping job')
            network_list.remove(network)
            continue
        with open(solver_path, 'r') as search:
            for line in search:
                line = line.rstrip()  # remove '\n' at end of line
                if line.startswith('snapshot_prefix: '):
                    snapshot_path = line.replace('snapshot_prefix: ', '').replace('"', '')
        if not os.path.exists(snapshot_path):
            generated_dirs.append(snapshot_path)
            os.makedirs(snapshot_path)
        else:
            log.error('tmp directory is not empty! Aborting')
            sys.exit()
    return network_list, generated_dirs

if __name__ == '__main__':
    args = get_args()
    if args.debug:
        log.setLevel(logging.DEBUG);
    else:
        log.setLevel(logging.INFO);
    # prepare folders
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    # create output folder for this run
    dir_name = datetime.datetime.now().strftime('%Y-%m-%d_%Hh-%Mm-%Ss') + '_experiment'
    os.makedirs(os.path.join(args.output_path, dir_name))
    fileHandler = logging.FileHandler(os.path.join(args.output_path, dir_name, 'auto_trainer.log'))
    fileHandler.setFormatter(logFormatter)
    log.addHandler(fileHandler)

    # load networks
    parsed_network_list = get_networks_from_file(args.jobs_file)
    # network_list += get_networks_from_python()
    parsed_network_list, tmp_dirs = generate_output_directories(parsed_network_list)
    log.info('Parsed {} job(s)'.format(parsed_network_list.__len__()))
    #for i in range(parsed_network_list.__len__()):
    #    job_path = os.path.join(args.output_path, 'tmp', 'job{}'.format(i))
    #    if not os.path.exists(job_path):
    #        os.makedirs(job_path)
    #    else:
    #        log.error('tmp directory is not empty! Aborting')
    #        sys.exit()
    #train_networks(parsed_network_list, os.path.join(args.output_path, dir_name))
    log.info('cleaning up tmp dir')
    # move all from tmp_dir to correct folder
    #for i in range(parsed_network_list.__len__()):
    #    job_path = os.path.join(args.output_path, 'tmp', 'job{}'.format(i))
    #    shutil.move(job_path, os.path.join(args.output_path, dir_name, 'weights'))
    #for path in tmp_dirs:
    #    shutil.move(path, os.path.join(args.output_path, dir_name, 'weights'))
    #os.rmdir(os.path.join(args.output_path, 'tmp'))
