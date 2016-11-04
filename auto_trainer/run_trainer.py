import datetime
import json
import logging
import os
import sys
import time
import configargparse
import shutil

from NNTrainClsSub import NetworkTrainer
from trainer_utils import check_job, generate_output_directory, draw_job_net,\
     draw_job_plot, generate_parsed_splitted_logs, extract_filters, draw_job_plot2

logFormatter = logging.Formatter("%(asctime)s [%(module)14s] [%(levelname)5s] %(message)s")
log = logging.getLogger()
log.setLevel(logging.DEBUG)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
log.addHandler(consoleHandler)


def get_networks_from_python():
    #with open('examples/mnist/lenet_auto_train.prototxt', 'w') as f:
    #    f.write(str(example_lenet('examples/mnist/mnist_train_lmdb', 64)))

    #with open('examples/mnist/lenet_auto_test.prototxt', 'w') as f:
    #    f.write(str(example_lenet('examples/mnist/mnist_test_lmdb', 100)))
    # todo make solver file
    return []


def get_next_job(jobs_file):
    jobs_dict = json.load(open(jobs_file, 'r'))
    log.info('fetching next job from {}'.format(jobs_file))
    for tmp_job in jobs_dict:
        if 'ignore' in jobs_dict[tmp_job]:
            if jobs_dict[tmp_job]['ignore']:
                continue
        if jobs_dict[tmp_job]['name'] not in worked_job_names:
            checked_job = check_job(jobs_dict[tmp_job])
            if checked_job is not None:
                log.info('jobs completed: {}, new job: {}'.format(worked_job_names.__len__(),
                                                                  jobs_dict[tmp_job]['name']))
                return checked_job, True
            else:
                log.error('check_job returned None for this job: {}'.format(jobs_dict[tmp_job]['name']))
        #else:
        #    log.info('already completed job with name: {}'.format(jobs_dict[tmp_job]['name']))
    #log.debug('No new jobs found, terminating')
    return None, False


def start_train_thread(job, run_on_gpu_num):
    job['gpu_num'] = run_on_gpu_num

    train_thread = NetworkTrainer(job, caffe_trainer_tool, caffe_tool_path, experiment_output_path)
    #train_thread.daemon = True
    train_thread.setName('{}'.format(job['name']))
    train_thread.start()

    return train_thread


def get_args():
    configpath = os.path.join(os.path.dirname(__file__), 'config.ini')
    parser = configargparse.ArgParser(default_config_files=[configpath])
    parser.add_argument('--jobs-file', type=str,
                        help='json file, gets parsed from trainingsjobs')
    parser.add_argument('--caffe-path', type=str,
                        help='path to caffe home dir')
    parser.add_argument('--gpu-count', type=int,
                        help='path to caffe home dir')
    parser.add_argument('--output-path', type=str,
                        help='Path, where auto_trainer will create an output-directory')
    parser.add_argument('--debug', help='Debug Mode', action='store_true')
    #parser.set_defaults(DEBUG=True)

    return parser.parse_args()


def check_if_no_files(path):
    list = os.listdir(path)
    for sub_path in list:
        if os.path.isfile(sub_path):
            return False
        else:
            if not check_if_no_files(sub_path):
                return False
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
    experiment_output_path = os.path.join(output_dir, dir_name)
    os.makedirs(experiment_output_path)
    fileHandler = logging.FileHandler(os.path.join(experiment_output_path, 'auto_trainer.log'))
    fileHandler.setFormatter(logFormatter)
    log.addHandler(fileHandler)
    log.info('output dir for this run:\n{}'.format(experiment_output_path))

    # check pathes from args
    caffe_trainer_tool = os.path.join(args.caffe_path, 'build', 'tools', 'caffe')
    if not os.path.isfile(caffe_trainer_tool):
        log.error('caffe_trainer_tool not found at:\n{}'.format(caffe_trainer_tool))
        sys.exit()
    caffe_tool_path = os.path.join(args.caffe_path, 'python')
    if not os.path.isdir(caffe_tool_path):
        log.error('caffe_tool_path not found at:\n{}'.format(caffe_tool_path))
        sys.exit()
    job_file_path = os.path.join(os.path.dirname(__file__), '..', args.jobs_file)
    if not os.path.isfile(job_file_path):
        log.error('jobfile not found at:\n{}'.format(job_file_path))
        sys.exit()
    log.info('using {} GPUs for training'.format(args.gpu_count))
    free_gpu_ids = [x for x in range(args.gpu_count)]
    # start training
    worked_job_names = []
    running_threads = []
    while True:

        # check on threads
        for thread in running_threads:
            if not thread.isAlive():
                job = thread.get_job()
                log.info('GPU {} Thread "{}" is finished, duration: {}'.format(job['gpu_num'],
                                                                               thread.getName(),
                                                                               job['duration']))
                running_threads.remove(thread)
                free_gpu_ids.append(job['gpu_num'])

        if free_gpu_ids.__len__() == 0:
            time.sleep(120)
            continue

        try:
            # load new job from jobs.json
            job, do_work = get_next_job(job_file_path)
        except ValueError:
            log.error('json not correctly formatted!')
            break;
        if not do_work:
            if running_threads.__len__() > 0:
                #log.debug('Couldn\'t find new job, but {} threads still running, sleeping'.format(running_threads.__len__()))
                time.sleep(120)
            else:
                job_names_formated = ''
                for name in worked_job_names:
                    job_names_formated += '{}\n'.format(name)
                log.info('Couldn\'t find new job and all jobs completed, ending script. '
                         'Completed jobs:\n{}'.format(job_names_formated))
                break
        else:
            # run training for job
            if free_gpu_ids.__len__() > 0:
                job['output_dir'], success = generate_output_directory(job['solver_path'],
                                                                       job['model_path'],
                                                                       job['snapshot_path'])
                if success:
                    running_threads.append(start_train_thread(job, free_gpu_ids[0]))
                    worked_job_names.append(job['name'])
                    free_gpu_ids.remove(free_gpu_ids[0])
                    time.sleep(2)  # just for better console readability
                    for train_thread in running_threads:
                        log.info('GPU {}: {}, started: {}'.format(train_thread.get_job()['gpu_num'],
                                                                  train_thread.get_job()['name'],
                                                                  train_thread.get_job()['start_time']))
                    time.sleep(3)  # just for better console readability
            else:
                #log.debug('all {} gpus working, sleeping'.format(args.gpu_count))
                time.sleep(120)



    log.info('all jobs completed')
    log.info('cleaning up tmp dir')
    if os.path.exists(os.path.join(output_dir, 'tmp')):
        if check_if_no_files(os.path.join(output_dir, 'tmp')):
            shutil.rmtree(os.path.join(output_dir, 'tmp'))
            #os.rmdir(os.path.join(output_dir, 'tmp'))
        else:
            log.error('there are still files in dir, please delete manually:\n'
                      '{}'.format(os.path.join(output_dir, 'tmp')))
