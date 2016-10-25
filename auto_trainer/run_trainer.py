import datetime
import json
import logging
import os
import shutil
import time
import sys
from NNTrainClsSub import NetworkTrainer
from trainer_utils import check_job, generate_output_directory, get_networks_from_file, \
    get_args, draw_job_net, draw_job_plot, generate_parsed_splitted_logs, extract_filters, draw_job_plot2

# global defines
#CAFFE_TOOL_PATH = '/home/ellerch/bin/caffe/python/'

logFormatter = logging.Formatter("%(asctime)s [%(module)14s] [%(levelname)5s] %(message)s")
log = logging.getLogger()
log.setLevel(logging.DEBUG)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
log.addHandler(consoleHandler)


def save_job_stats_to_json(job):
    log.info('Job "{}" completed in {}. Accuracy: {:.3f}'.format(job['name'],
                                                                 job['duration'],
                                                                 job['accuracy']))
    log_path = os.path.join(job['snapshot_path'], job['name'].replace(' ', '_') + "_stats.json")
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


def get_next_job(jobs_file):
    jobs_dict = json.load(open(jobs_file, 'r'))
    log.info('fetching next job from {}'.format(jobs_file))
    for tmp_job in jobs_dict:
        if jobs_dict[tmp_job]['ignore']:
            log.info('ignoring job: ' + jobs_dict[tmp_job]['name'])
            continue
        if jobs_dict[tmp_job]['name'] not in finished_job_names:
            checked_job = check_job(jobs_dict[tmp_job])
            if checked_job is not None:
                log.info('jobs completed: {}, new job: {}'.format(finished_job_names.__len__(),
                                                                  jobs_dict[tmp_job]['name']))
                return checked_job, True
            else:
                log.error('check_job returned None for this job: {}'.format(jobs_dict[tmp_job]['name']))
        #else:
        #    log.info('already completed job with name: {}'.format(jobs_dict[tmp_job]['name']))
    #log.debug('No new jobs found, terminating')
    return None, False


def train_network(job):
    log.info('training starts for job {}'.format(job['name']))
    job['start_time'] = datetime.datetime.now().strftime('%Y-%m-%d_%Hh-%Mm-%Ss')
    train_thread = NetworkTrainer(job, log)
    train_thread.daemon = True
    train_thread.setName('{}'.format(job['name']))
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
                     os.path.join(job['snapshot_path'], job['name'] + '_net.png'), '/home/ellerch/bin/caffe/python/')

        # training is done, write log and other output
        generate_parsed_splitted_logs(job['caffe_log_path'],
                                      job['snapshot_path'])

        draw_job_plot(job['caffe_log_path'],
                      os.path.join(job['snapshot_path'], job['name'] + '_training_plot.png'))

        draw_job_plot2(os.path.join(job['snapshot_path'], "parsed_caffe_log.test"),
                       os.path.join(job['snapshot_path'], job['name'] + '_better_training_plot.png'))
        acc, training_loss = get_avg_acc_and_loss(os.path.join(job['snapshot_path'], "parsed_caffe_log.test"))
        thread_stats = train_thread.get_stats()
        job['accuracy'] = acc
        job['test_loss'] = training_loss
        job['duration'] = thread_stats['duration_str']

        # get best caffemodel
        weights_path = get_best_caffemodel(job['snapshot_path'])
        if weights_path is not None:
            extract_filters(job['model_path'], weights_path, job['snapshot_path'])
        else:
            log.error('Could not find a caffemodel for solver in {}, '
                      'no filters extracted.'.format(job['snapshot_path']))
        save_job_stats_to_json(job)

    except KeyboardInterrupt:
        log.info('KeyboardInterrupt, stopping current job')
        return '0s', False
    except SystemExit:
        log.error('SystemExit, stopping script')
        raise
    except:
        log.error("Unexpected error during training, processing next job")
        log.error(sys.exc_info())
        return train_thread.get_duration(), False

    return job['duration'], True


def move_all_files_from_to(src_path, dest_path):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    else:
        log.error('there already is a completed job with name {} in output dir!'.format(dest_path))
        return False
    file_list = os.listdir(src_path)
    log.info('moving all files from\n{}\nto\n{}'.format(src_path, dest_path))
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


if __name__ == '__main__':
    args = get_args()
    if args.debug:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)
    # prepare folders
    output_dir = os.path.join(os.path.dirname(__file__), '..', args.output_path)
    if not os.path.exists(output_dir):
        log.info('output dir does not exist, creating:\n{}'.format(output_dir))
        os.makedirs(output_dir)
    # create output folder for this run
    dir_name = datetime.datetime.now().strftime('%Y-%m-%d_%Hh-%Mm-%Ss') + '_experiment'
    output_path = os.path.join(output_dir, dir_name)
    os.makedirs(output_path)
    fileHandler = logging.FileHandler(os.path.join(output_path, 'auto_trainer.log'))
    fileHandler.setFormatter(logFormatter)
    log.addHandler(fileHandler)
    log.info('output dir for this run:\n{}'.format(output_path))

    job_file_path = os.path.join(os.path.dirname(__file__), '..', args.jobs_file)
    if not os.path.isfile(job_file_path):
        log.error('jobfile not found at:\n{}'.format(job_file_path))
        sys.exit()

    finished_job_names = []
    while True:
        # load new job from jobs.json
        job, do_work = get_next_job(job_file_path)
        if not do_work:
            log.info('Couldn\'t find new job, ending script')
            break
        job['output_dir'] = generate_output_directory(job['solver_path'],
                                                      job['model_path'],
                                                      job['snapshot_path'])

        # run training for job
        duration, completed = train_network(job)
        job['job_duration'] = duration
        job['completed'] = completed
        finished_job_names.append(job['name'])

        # cleanup
        # after training, post stats
        if job['completed']:
            move_all_files_from_to(job['snapshot_path'], os.path.join(output_path, job['name']))
            log.info('Job {}: completed in {}'.format(job['name'], job['job_duration']))
        else:
            move_all_files_from_to(job['snapshot_path'], os.path.join(output_path, job['name']+'_failed'))
            log.info('Job {}: failed in {}'.format(job['name'], job['job_duration']))

    log.info('all jobs completed')
    log.info('cleaning up tmp dir')
    if os.path.exists(os.path.join(output_dir, 'tmp')):
        os.rmdir(os.path.join(output_dir, 'tmp'))
