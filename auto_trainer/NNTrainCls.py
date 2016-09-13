import sys
import caffe
import logging
from threading import Thread
from timeit import default_timer as timer


class NetworkTrainer2(Thread):
    def __init__(self, job_file, log):
        Thread.__init__(self)
        self.job_file = job_file
        self.log = log
        self.train_duration = None
        self.test_duration = None
        self.accuracy = 0.0

    def run(self):
        self.log.debug('Name: {}, my job file: {}'.format(self.getName(), self.job_file))

        start = timer()
        caffe.set_mode_gpu()
        caffe.set_device(0)

        # By default it is the SGD solver
        # Optimizes the parameters of a Net using stochastic gradient descent (SGD) with momentum.
        solver = caffe.get_solver(self.job_file)
        # subprocess.call(".build/tools/caffe train You/solver/path 2>&1 | tee You/log/path", shell=True)
        solver.solve()
        solver.net
        end = timer()
        self.train_duration = end - start
        self.log.info('training finished, duration: {}'.format(self.train_duration))

        start = timer()
        accuracy = 0
        Xt = 100
        batch_size = solver.test_nets[0].blobs['data'].num
        test_iters = int(Xt / batch_size)
        for i in xrange(test_iters):
            solver.test_nets[0].forward()
            accuracy += solver.test_nets[0].blobs['accuracy'].data
        accuracy /= test_iters
        end = timer()
        self.test_duration = end - start
        self.accuracy = accuracy
        # The higher the accuracy, the better !
        self.log.info('testing finished, accuracy: {:.3f}'.format(accuracy))

    def get_stats(self):
        minutes, sec = divmod(self.train_duration + self.test_duration, 60)
        hours, minutes = divmod(minutes, 60)
        return {'duration': (self.train_duration + self.test_duration),
                'duration_str': '%02dh %02dm %02ds' % (hours, minutes, sec),
                'train_duration': str(self.train_duration),
                'test_duration': self.test_duration,
                'name': self.getName(),
                'accuracy': self.accuracy}

    def get_duration(self):
        return self.train_duration + self.test_duration

