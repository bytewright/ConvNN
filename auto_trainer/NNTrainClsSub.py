import sys
import caffe
import logging
import time
import os
import subprocess
from threading import Thread
from timeit import default_timer as timer

caffeTool = '/home/ellerch/bin/caffe/build/tools/caffe'  # path to caffe script


class NetworkTrainer(Thread):
    def __init__(self, job, log):
        Thread.__init__(self)
        self.job = job
        self.log = log
        self.train_duration = None
        self.test_duration = None
        self.accuracy = 0.0
        self.returncode = None

    def run(self):
        self.log.info('Thread start\nName: {}\njob file: {}'.format(self.getName(), self.job['solver_path']))
        logfile = open(os.path.join(self.job['snapshot_path'], "tmp_caffe_training.log"), "w")
        start = timer()
        # output = '{} train -solver {} {}\n\n'.format(caffeTool, self.job_file, '-gpu 0')
        solver_process = subprocess.Popen([caffeTool, 'train', '-solver', self.job['solver_path'], '-gpu', '0'],
                                          stdout=logfile,
                                          stderr=subprocess.STDOUT)
        solver_process.communicate()
        end = timer()
        #self.returncode = solver_process.returncode
        while self.returncode is None:
            time.sleep(2)
            self.returncode = solver_process.returncode
        if self.returncode is not 0:
            self.log.error('solver return code: ' + str(solver_process.returncode))
        else:
            self.log.info('solver exited successfully (code {})'.format(self.returncode))
        self.train_duration = end - start
        self.log.info('training finished, duration: {:.2f}s'.format(self.train_duration))
        os.rename(os.path.join(self.job['snapshot_path'], "tmp_caffe_training.log"),
                  os.path.join(self.job['snapshot_path'], "caffe_training.log"))
        #with open(os.path.join(self.job_output_dir, "caffe_training.log"), "w") as text_file:
        #    text_file.write(output)
        # todo load net, get conv-filter

    def get_caffe_log_path(self):
        return os.path.join(self.job['snapshot_path'], "caffe_training.log")

    def get_stats(self):
        minutes, sec = divmod(self.train_duration, 60)
        hours, minutes = divmod(minutes, 60)
        return {'duration': self.train_duration,
                'duration_str': '%02dh %02dm %02ds' % (hours, minutes, sec),
                'train_duration': str(self.train_duration),
                'test_duration': 0,
                'name': self.getName(),
                'accuracy': 0}

    def get_duration(self):
        return self.train_duration

    def get_training_returncode(self):
        return self.returncode
