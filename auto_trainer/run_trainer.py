import datetime
import json
import logging
import os
import shutil
import time

from NNTrainClsSub import NetworkTrainer
from trainer_utils import generate_output_directory, get_networks_from_file, \
    get_args, draw_job_net, draw_job_plot, generate_parsed_splitted_logs

# global defines
CAFFE_TOOL_PATH = '/home/ellerch/bin/caffe/python/'

logFormatter = logging.Formatter("%(asctime)s [%(module)14s] [%(levelname)5s] %(message)s")
log = logging.getLogger()
log.setLevel(logging.DEBUG)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
log.addHandler(consoleHandler)


def generate_job_log(output_dir, job_index, stats_dict):
    log.info('Job "{}" completed in {:.2f}s. Accuracy: {:.3f}'.format(stats_dict['name'],
                                                                      stats_dict['duration'],
                                                                      stats_dict['accuracy']))
    log_path = os.path.join(output_dir, "stats.json".format(job_index))
    json.dump(stats_dict, open(log_path, 'w'))
    return


def get_networks_from_python():
    #with open('examples/mnist/lenet_auto_train.prototxt', 'w') as f:
    #    f.write(str(example_lenet('examples/mnist/mnist_train_lmdb', 64)))

    #with open('examples/mnist/lenet_auto_test.prototxt', 'w') as f:
    #    f.write(str(example_lenet('examples/mnist/mnist_test_lmdb', 100)))
    # todo make solver file
    return []


def get_avg_acc_and_loss(log_path):
    delimiter = ' '
    test_data = []
    with open(log_path, 'r') as dest_f:
        for line in dest_f.readlines():
            line = line.rstrip()
            test_data.append([data for data in line.split(delimiter) if data is not ''])
    avg_acc = 0.0
    avg_loss = 0.0
    use_last_n = 10.0
    for data in test_data[-int(use_last_n):]:
        if len(data) < 4:
            log.error('not enough data points in test log!')
            continue
        avg_acc += float(data[2]) / use_last_n
        avg_loss += float(data[3]) / use_last_n
    return avg_acc, avg_loss


def train_networks(network_list, output_path):
    train_threads = []
    for job in network_list:
        log.debug('reading job desc: ' + job)
        if job.startswith('#'):
            log.info('skipping comment: ' + job)
            continue

        solver_prototxt = job + 'solver.prototxt'
        if not open(solver_prototxt, 'r'):
            log.error('can\'t open solver.prototxt!, skipping job')
            continue

        jobID = network_list.index(job)
        # todo dir handling
        job_output_dir = os.path.join(output_path, 'job{}'.format(job[-2]))
        log.debug('creating output dir:\n{}'.format(job_output_dir))
        os.makedirs(job_output_dir)

        log.info('starting thread for job {}'.format(jobID))
        train_thread = NetworkTrainer(job, job_output_dir, solver_prototxt, log)
        train_threads.append(train_thread)
        train_thread.daemon = True
        train_thread.setName('thread {}'.format(jobID))
        train_thread.start()

        # generate image of NN
        time.sleep(2)  # wait for trainer to get started
        draw_job_net(solver_prototxt,
                     os.path.join(job_output_dir, 'net.png'), log)
        train_thread.join()

        # training is done, write log and other output
        generate_parsed_splitted_logs(os.path.join(job_output_dir, "caffe_training.log"), job_output_dir, log)
        draw_job_plot(os.path.join(job_output_dir, "caffe_training.log"), log)
        acc, training_loss = get_avg_acc_and_loss(os.path.join(job_output_dir, "parsed_caffe_log.test"))
        thread_stats = train_thread.get_stats()
        thread_stats['accuracy'] = acc
        thread_stats['test_loss'] = training_loss
        generate_job_log(job_output_dir, jobID, thread_stats)

    log.info('all jobs completed')
    for job in train_threads:
        minutes, sec = divmod(job.get_duration(), 60)
        hours, minutes = divmod(minutes, 60)
        log.info('thread {}: completed in {}'.format(job.getName(), '%02dh %02dm %02ds' % (hours, minutes, sec)))


if __name__ == '__main__':
    args = get_args()
    if args.debug:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)
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
    parsed_network_list = get_networks_from_file(args.jobs_file, log)
    # network_list += get_networks_from_python()
    tmp_dirs = []
    for network in parsed_network_list:
        new_dir = generate_output_directory(network, log)
        if new_dir is not None:
            tmp_dirs.append(new_dir)
        else:
            parsed_network_list.remove(network)

    log.info('Parsed {} job(s)'.format(parsed_network_list.__len__()))
    train_networks(parsed_network_list, os.path.join(args.output_path, dir_name))

    log.info('cleaning up tmp dir')
    for path in tmp_dirs:
        shutil.move(path, os.path.join(args.output_path, dir_name, path.split('/')[-2]))
    if os.path.exists(os.path.join(args.output_path, 'tmp')):
        os.rmdir(os.path.join(args.output_path, 'tmp'))
