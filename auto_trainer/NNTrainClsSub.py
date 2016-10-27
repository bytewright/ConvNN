import sys
import caffe
import logging
import time
import os
import subprocess
from threading import Thread
from timeit import default_timer as timer

#caffeTool = '/home/ellerch/bin/caffe/build/tools/caffe'  # path to caffe script


class NetworkTrainer(Thread):
    def __init__(self, job, caffe_trainer_path):
        Thread.__init__(self)
        self.job = job
        self.caffe_trainer_path = caffe_trainer_path
        self.train_duration = None
        self.test_duration = None
        self.accuracy = 0.0
        self.returncode = None

    def run(self):
        logging.info('Thread start\nName: {}\njob file: {}'.format(self.getName(), self.job['solver_path']))
        logfile = open(os.path.join(self.job['snapshot_path'], "tmp_caffe_training.log"), "w")
        start = timer()
        # output = '{} train -solver {} {}\n\n'.format(caffeTool, self.job_file, '-gpu 0')
        solver_process = subprocess.Popen([self.caffe_trainer_path, 'train',
                                           '-solver', self.job['solver_path'],
                                           '-gpu', 'all'],
                                          stdout=logfile,
                                          stderr=subprocess.STDOUT)
        try:
            solver_process.communicate()
        except KeyboardInterrupt:
            logging.info('KeyboardInterrupt, stopping current job')
        end = timer()

        while self.returncode is None:
            time.sleep(2)
            self.returncode = solver_process.returncode
        if self.returncode is not 0:
            logging.error('solver return code: ' + str(solver_process.returncode))
        else:
            logging.info('solver exited successfully (code {})'.format(self.returncode))
        self.train_duration = end - start
        logging.info('training finished, duration: {:.2f}s'.format(self.train_duration))
        os.rename(os.path.join(self.job['snapshot_path'], "tmp_caffe_training.log"),
                  os.path.join(self.job['snapshot_path'], "caffe_training.log"))

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
