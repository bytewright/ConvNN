import datetime
import json
import logging
import os
import shutil
import time
import sys
from NNTrainClsSub import NetworkTrainer
from trainer_utils import check_job, generate_output_directory, get_networks_from_file, \
    get_args, draw_job_net, draw_job_plot, generate_parsed_splitted_logs, extract_filters

# global defines
CAFFE_TOOL_PATH = '/home/ellerch/bin/caffe/python/'

logFormatter = logging.Formatter("%(asctime)s [%(module)14s] [%(levelname)5s] %(message)s")
log = logging.getLogger()
log.setLevel(logging.DEBUG)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
log.addHandler(consoleHandler)


def generate_job_log(job):
    log.info('Job "{}" completed in {}. Accuracy: {:.3f}'.format(job['name'],
                                                                 job['duration'],
                                                                 job['accuracy']))
    log_path = os.path.join(job['snapshot_path'], job['name'], "_stats.json")
    json.dump(job, open(log_path, 'w'), sort_keys=True, indent=4, separators=(',', ': '))
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
    if test_data.__len__() < 10:
        log.info('found only {} lines in log'.format(test_data.__len__()))
        use_last_n = 1.0
    else:
        use_last_n = 10.0
    avg_acc = 0.0
    avg_loss = 0.0
    found_data_points = 0
    for data in reversed(test_data):
        if found_data_points == use_last_n:
            break
        if len(data) < 4:
            continue
        avg_acc += float(data[2]) / use_last_n
        avg_loss += float(data[3]) / use_last_n
        found_data_points += 1
    return avg_acc, avg_loss


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


def train_network(job):
    log.info('starting training for job {}'.format(job['name']))
    job['start_time'] = datetime.datetime.now().strftime('%Y-%m-%d_%Hh-%Mm-%Ss')
    train_thread = NetworkTrainer(job, log)
    train_thread.daemon = True
    train_thread.setName('{}-thread'.format(job['name']))
    train_thread.start()

    train_thread.join()
    if train_thread.get_training_returncode() is 0:
        log.info('training finished, processing data from log')
        job['caffe_log_path'] = train_thread.get_caffe_log_path()
    else:
        log.error('training was not successful! skipping data processing')
        return '0', False
    # all data processing for jobs
    try:
        # generate image of NN
        draw_job_net(job['solver_path'],
                     os.path.join(job['snapshot_path'], job['name'] + '_net.png'), log)

        # training is done, write log and other output
        generate_parsed_splitted_logs(job['caffe_log_path'],
                                      job['snapshot_path'], log)

        draw_job_plot(job['caffe_log_path'],
                      os.path.join(job['snapshot_path'], job['name'] + '_training_plot.png'),
                      log)
        acc, training_loss = get_avg_acc_and_loss(os.path.join(job['snapshot_path'], "parsed_caffe_log.test"))
        thread_stats = train_thread.get_stats()
        job['accuracy'] = acc
        job['test_loss'] = training_loss
        job['duration'] = thread_stats['duration_str']

        # get best caffemodel
        weights_path = get_best_caffemodel(job['snapshot_path'])
        if weights_path is not None:
            extract_filters(job['solver_path'], weights_path, job['snapshot_path'], log)
        else:
            log.error('Could not find a caffemodel for solver in {}, '
                      'no filters extracted.'.format(job['snapshot_path']))
        generate_job_log(job)

    except (KeyboardInterrupt, SystemExit):
        log.info('KeyboardInterrupt, raising error')
        raise
    except:
        log.error("Unexpected error, processing next job")
        log.error(sys.exc_info())
        return train_thread.get_duration(), False

    return train_thread.get_duration(), True


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

    # load jobs from jobs.json
    jobs_dict = json.load(open(args.jobs_file, 'r'))
    jobs = []
    for tmp_job in jobs_dict:
        checked_job = check_job(jobs_dict[tmp_job], log)
        if checked_job is not None:
            generate_output_directory(checked_job['solver_path'],
                                      checked_job['model_path'],
                                      checked_job['snapshot_path'],
                                      log)
            jobs.append(checked_job)
    log.info('Parsed {} job(s)'.format(jobs.__len__()))

    # run training for each job
    for i in range(jobs.__len__()):
        duration, completed = train_network(jobs[i])
        jobs[i]['job_duration'] = duration
        jobs[i]['completed'] = completed
    log.info('all jobs completed')

    # after training, post stats
    for tmp_job in jobs:
        minutes, sec = divmod(tmp_job['job_duration'], 60)
        hours, minutes = divmod(minutes, 60)
        if jobs[i]['completed']:
            log.info('Job {}: completed in {}'.format(tmp_job['name'], '%02dh %02dm %02ds' % (hours, minutes, sec)))
        else:
            log.info('Job {}: failed in {}'.format(tmp_job['name'], '%02dh %02dm %02ds' % (hours, minutes, sec)))

    # clean up
    log.info('cleaning up tmp dir')
    for tmp_job in jobs:
        output_path = os.path.join(args.output_path, dir_name)
        shutil.move(tmp_job['snapshot_path'], output_path)
    if os.path.exists(os.path.join(args.output_path, 'tmp')):
        os.rmdir(os.path.join(args.output_path, 'tmp'))
