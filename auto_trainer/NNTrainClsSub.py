import datetime
import logging
import time
import sys
import os
import subprocess
from threading import Thread
from timeit import default_timer as timer
from trainer_utils import save_job_stats_to_json, get_best_caffemodel, draw_job_net,\
     generate_parsed_splitted_logs, extract_filters, get_training_stats, \
     move_all_files_from_to, generate_parsed_splitted_logs2, draw_job_plot


class NetworkTrainer(Thread):
    def __init__(self, job, caffe_trainer_path, caffe_tool_path, experiment_output_path):
        Thread.__init__(self)
        self.job = job
        self.job['start_time'] = datetime.datetime.now().strftime('%Y-%m-%d_%Hh-%Mm-%Ss')
        self.caffe_trainer_path = caffe_trainer_path
        self.caffe_tool_path = caffe_tool_path
        self.experiment_output_path = experiment_output_path

    def run(self):
        self.log_info('Thread start\nName: {}\njob file: {}'.format(self.getName(), self.job['solver_path']))
        logfile = open(os.path.join(self.job['snapshot_path'], "tmp_caffe_training.log"), "w")
        start = timer()
        # start subprocess for training
        solver_process = subprocess.Popen([self.caffe_trainer_path, 'train',
                                           '-solver', self.job['solver_path'],
                                           '-gpu', str(self.job['gpu_num'])],
                                          stdout=logfile,
                                          stderr=subprocess.STDOUT)
        try:
            solver_process.communicate()
        except KeyboardInterrupt:
            self.log_info('KeyboardInterrupt, stopping current job')
            solver_process.terminate()
        end = timer()

        returncode = None
        while returncode is None:
            time.sleep(2)
            returncode = solver_process.returncode
        if returncode is not 0:
            self.job['completed'] = False
            self.log_error('solver return code: {}, check caffe logfile.'.format(solver_process.returncode))
        else:
            self.log_info('solver exited successfully (code {})'.format(returncode))

        self.log_info('training finished, duration: {:.2f}s'.format(end - start))
        os.rename(os.path.join(self.job['snapshot_path'], "tmp_caffe_training.log"),
                  os.path.join(self.job['snapshot_path'], "caffe_training.log"))

        # start analyse toolchain
        minutes, sec = divmod(end - start, 60)
        hours, minutes = divmod(minutes, 60)
        self.job['duration'] = '%02dh %02dm %02ds' % (hours, minutes, sec)
        self.job['caffe_log_path'] = os.path.join(self.job['snapshot_path'], "caffe_training.log")
        if returncode is 0:
            try:
                # generate image of NN
                draw_job_net(self.job['solver_path'],
                             os.path.join(self.job['snapshot_path'], self.job['name'] + '_net.png'), self.caffe_tool_path)

                # training is done, write log and other output
                # generates csv file
                generate_parsed_splitted_logs2(self.job['caffe_log_path'],
                                               os.path.join(self.job['snapshot_path'],
                                                            self.job['name'] + '_caffe_test_log.csv'))
                # generates .test and .train file
                generate_parsed_splitted_logs(self.job['caffe_log_path'],
                                              self.job['snapshot_path'])

                # parse .test logfile and average over last items
                stats_dict = get_training_stats(os.path.join(self.job['snapshot_path'],
                                                             self.job['name'] + '_caffe_test_log.csv'))
                self.job.update(stats_dict)
                #self.job['accuracy'], self.job['test_loss'] = get_avg_acc_and_loss(os.path.join(
                #                                                                    self.job['snapshot_path'],
                #                                                                    "parsed_caffe_log.test"))
                # uses data from csv file to generate plots
                draw_job_plot(os.path.join(self.job['snapshot_path'], self.job['name'] + '_caffe_test_log.csv'),
                              os.path.join(self.job['snapshot_path'], self.job['name'] + '_training_plot2.png'))

                # parse dir for latest caffemodel
                weights_path = get_best_caffemodel(self.job['snapshot_path'])
                if weights_path is not None:
                    # extract all convolution filters from network
                    extract_filters(self.job['model_path'], weights_path, self.job['snapshot_path'])
                else:
                    self.log_error('Could not find a caffemodel for solver in {}, '
                                  'no filters extracted.'.format(self.job['snapshot_path']))

                # generates json file from job dict
                save_job_stats_to_json(self.job)
                self.job['completed'] = True
            except KeyboardInterrupt:
                self.log_info('KeyboardInterrupt, stopping current job')
                self.job['completed'] = False
            except SystemExit:
                self.log_error('SystemExit, stopping script')
                raise
            except:
                self.log_error("Unexpected error during training")
                self.log_error(sys.exc_info())
                self.job['completed'] = False
        # cleanup
        # after training, post stats
        if self.job['completed']:
            # moves everything from snapshot path to experiment dir, given by run_trainer
            move_all_files_from_to(self.job['snapshot_path'], os.path.join(self.experiment_output_path, self.job['name']))
            self.log_info('Job {}: completed in {}'.format(self.job['name'], self.job['duration']))
        else:
            # moves everything from snapshot path to experiment dir, given by run_trainer
            move_all_files_from_to(self.job['snapshot_path'], os.path.join(self.experiment_output_path, self.job['name'] + '_failed'))
            self.log_info('Job {}: failed in {}'.format(self.job['name'], self.job['duration']))

    # log functions for convenience
    def log_info(self, log_str):
        logging.info('(GPU {}):{}'.format(self.job['gpu_num'], log_str))

    def log_error(self, log_str):
        logging.error('(GPU {}):{}'.format(self.job['gpu_num'], log_str))

    def get_job(self):
        return self.job
