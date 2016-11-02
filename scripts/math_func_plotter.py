#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import pylab

def step(iter):
    base_lr = 0.01
    gamma = 0.1
    stepsize = 10000.0
    return base_lr * math.pow(gamma, math.floor(iter / stepsize))


def multistep_7_0(iter):
    base_lr = 0.01
    gamma = 0.1
    steps = [(25000,1),(40000,2),(55000,3),(70000,4),(80000,5),(95000,6),(105000,7)]
    for step in reversed(steps):
        if iter >= step[0]:
            #print '{} bigger than {}'.format(iter, step[0])
            return base_lr * math.pow(gamma, step[1])
    return base_lr

def multistep_6_1(iter):
    base_lr = 0.01
    gamma = 0.1
    steps = [(30000,1),(50000,2),(65000,3),(80000,4),(95000,5),(105000,6)]
    for step in reversed(steps):
        if iter >= step[0]:
            #print '{} bigger than {}'.format(iter, step[0])
            return base_lr * math.pow(gamma, step[1])
    return base_lr

def multistep_3(iter):
    base_lr = 0.01
    gamma = 0.1
    steps = [(25000,1),(60000,2),(95000,3)]

    for step in reversed(steps):
        if iter >= step[0]:
            #print '{} bigger than {}'.format(iter, step[0])
            return base_lr * math.pow(gamma, step[1])
    return base_lr


def inv_3(iter):
    base_lr = 0.01
    gamma = 0.0001
    power = 0.75
    return base_lr * math.pow(1 + gamma * iter, -power)


def poly_4(iter, power):
    base_lr = 0.01
    return base_lr * math.pow(1 - iter / max_iter, power)


def sigmoid(iter):
    base_lr = 0.01
    gamma = -1e-04
    stepsize = 20000.0
    return base_lr * (1.0 / (1.0 + math.exp(-gamma * (iter - stepsize))))


def exp(iter):
    base_lr = 0.01
    gamma = 0.9999
    return base_lr * math.pow(gamma, iter)

max_iter = 60000.0
lr_rate = 0.01
iters = np.arange(0, max_iter, 100)
lines = []

fig, ax1 = plt.subplots()


#lines.append(ax1.plot(iters, [step(x) for x in iters], label='step 10k'))
#lines.append(ax1.plot(iters, [sigmoid(x) for x in iters], label='sigmoid 10k'))
#lines.append(ax1.plot(iters, [multistep_3(x) for x in iters], label='multistep 3'))
#lines.append(ax1.plot(iters, [inv_3(x) for x in iters], label='inv'))
lines.append(ax1.plot(iters, [exp(x) for x in iters], label='exp'))
#lines.append(ax1.plot(iters, [poly_4(x, 2.0) for x in iters], label='poly 2'))
#lines.append(ax1.plot(iters, [poly_4(x, 0.5) for x in iters], label='poly 0.5'))
#lines.append(ax1.plot(iters, [poly_4(x, 1.0) for x in iters], label='poly 1'))
plt.ylim([-0.001, 0.015])
plt.xlim([0, max_iter+1000])
ax1.set_xlabel('Iterationen')
ax1.set_ylabel('Learningrate')

#legend = plt.legend(handles=[x[0] for x in lines], bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#               ncol=2, mode="expand", borderaxespad=0.)
legend = plt.legend(handles=[x[0] for x in lines], bbox_to_anchor=(1.05, 1), loc=2,
               borderaxespad=0.)
plt.show()
fig.savefig('test.png', bbox_extra_artists=(legend,), bbox_inches='tight')