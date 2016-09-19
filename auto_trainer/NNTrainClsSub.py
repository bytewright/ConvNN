import sys
import caffe
import logging
import os
import subprocess
from threading import Thread
from timeit import default_timer as timer

caffeTool = '/home/ellerch/bin/caffe/build/tools/caffe'  # path to caffe script


class NetworkTrainer(Thread):
    def __init__(self, job_path, job_output_dir, job_file, log):
        Thread.__init__(self)
        self.job_file = job_file
        self.job_output_dir = job_output_dir
        self.job_path = job_path
        self.log = log
        self.train_duration = None
        self.test_duration = None
        self.accuracy = 0.0

    def run(self):
        self.log.debug('Name: {}, my job file: {}'.format(self.getName(), self.job_file))
        logfile = open(os.path.join(self.job_output_dir, "tmp_caffe_training.log"), "w")
        start = timer()
        output = '{} train -solver {} {}\n\n'.format(caffeTool, self.job_file, '-gpu 0')
        solver_process = subprocess.Popen([caffeTool, 'train', '-solver', self.job_file, '-gpu', '0'],
                                          stdout=logfile,
                                          stderr=subprocess.STDOUT)
        output += solver_process.communicate()[0]
        end = timer()
        self.train_duration = end - start
        self.log.info('training finished, duration: {:.2f}s'.format(self.train_duration))

        with open(os.path.join(self.job_output_dir, "caffe_training.log"), "w") as text_file:
            text_file.write(output)
        # todo parse output for final accuracy and duration
        # todo load net, get conv-filter

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
