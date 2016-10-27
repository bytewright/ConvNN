import datetime
import logging
import time
import sys
import os
import subprocess
from threading import Thread
from timeit import default_timer as timer
from trainer_utils import get_avg_acc_and_loss, save_job_stats_to_json, get_best_caffemodel, draw_job_net,\
     generate_parsed_splitted_logs, extract_filters, draw_job_plot2, move_all_files_from_to


class NetworkTrainer(Thread):
    def __init__(self, job, caffe_trainer_path, caffe_tool_path, experiment_output_path):
        Thread.__init__(self)
        self.job = job
        self.caffe_trainer_path = caffe_trainer_path
        self.caffe_tool_path = caffe_tool_path
        self.experiment_output_path = experiment_output_path

    def run(self):
        self.log_info('Thread start\nName: {}\njob file: {}'.format(self.getName(), self.job['solver_path']))
        logfile = open(os.path.join(self.job['snapshot_path'], "tmp_caffe_training.log"), "w")
        self.job['start_time'] = datetime.datetime.now().strftime('%Y-%m-%d_%Hh-%Mm-%Ss')
        start = timer()
        # output = '{} train -solver {} {}\n\n'.format(caffeTool, self.job_file, '-gpu 0')
        solver_process = subprocess.Popen([self.caffe_trainer_path, 'train',
                                           '-solver', self.job['solver_path'],
                                           '-gpu', self.job['gpu_num']],
                                          stdout=logfile,
                                          stderr=subprocess.STDOUT)
        try:
            solver_process.communicate()
        except KeyboardInterrupt:
            self.log_info('KeyboardInterrupt, stopping current job')
        end = timer()

        returncode = None
        while returncode is None:
            time.sleep(2)
            returncode = solver_process.returncode
        if returncode is not 0:
            self.log_error('solver return code: ' + str(solver_process.returncode))
        else:
            self.log_info('solver exited successfully (code {})'.format(returncode))

        self.log_info('training finished, duration: {:.2f}s'.format(end - start))
        os.rename(os.path.join(self.job['snapshot_path'], "tmp_caffe_training.log"),
                  os.path.join(self.job['snapshot_path'], "caffe_training.log"))

        # start toolchain
        minutes, sec = divmod(end - start, 60)
        hours, minutes = divmod(minutes, 60)
        self.job['duration'] = '%02dh %02dm %02ds' % (hours, minutes, sec)
        self.job['caffe_log_path'] = os.path.join(self.job['snapshot_path'], "caffe_training.log")

        try:
            # generate image of NN
            draw_job_net(self.job['solver_path'],
                         os.path.join(self.job['snapshot_path'], self.job['name'] + '_net.png'), self.caffe_tool_path)

            # training is done, write log and other output
            generate_parsed_splitted_logs(self.job['caffe_log_path'],
                                          self.job['snapshot_path'])

            self.job['accuracy'], self.job['test_loss'] = get_avg_acc_and_loss(os.path.join(self.job['snapshot_path'],
                                                                                            "parsed_caffe_log.test"))

            draw_job_plot2(os.path.join(self.job['snapshot_path'], "parsed_caffe_log.test"),
                           os.path.join(self.job['snapshot_path'], self.job['name'] + '_better_training_plot.png'))

            # get best caffemodel
            weights_path = get_best_caffemodel(self.job['snapshot_path'])
            if weights_path is not None:
                extract_filters(self.job['model_path'], weights_path, self.job['snapshot_path'])
            else:
                self.log_error('Could not find a caffemodel for solver in {}, '
                              'no filters extracted.'.format(self.job['snapshot_path']))
            save_job_stats_to_json(self.job)

        except KeyboardInterrupt:
            self.log_info('KeyboardInterrupt, stopping current job')
            return '0s', False
        except SystemExit:
            self.log_error('SystemExit, stopping script')
            raise
        except:
            self.log_error("Unexpected error during training, processing next job")
            self.log_error(sys.exc_info())
        # cleanup
        # after training, post stats
        if self.job['completed']:
            move_all_files_from_to(self.job['snapshot_path'], os.path.join(self.experiment_output_path, self.job['name']))
            self.log_info('Job {}: completed in {}'.format(self.job['name'], self.job['job_duration']))
        else:
            move_all_files_from_to(self.job['snapshot_path'], os.path.join(self.experiment_output_path, self.job['name'] + '_failed'))
            self.log_info('Job {}: failed in {}'.format(self.job['name'], self.job['job_duration']))

    def log_info(self, log_str):
        logging.info('(GPU {}):{}'.format(self.job['gpu_num'], log_str))

    def log_error(self, log_str):
        logging.error('(GPU {}):{}'.format(self.job['gpu_num'], log_str))

    def get_job(self):
        return self.job
