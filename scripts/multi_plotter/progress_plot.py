import os
import configargparse
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pylab


def load_data(data_file):
    delimiter = ' '
    test_data = []
    with open(data_file, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if line.split(delimiter).__len__() < 3:
                print('incomplete line found')
            else:
                test_data.append([data for data in line.split(delimiter) if data is not ''])

    return test_data[1:]


if __name__ == "__main__":

    configpath = os.path.join(os.path.dirname(__file__), 'plotter_config.ini')
    parser = configargparse.ArgParser(default_config_files=[configpath])
    parser.add_argument('--reference_data', help='path to reference test data, generated by caffe parse_log.py')
    parser.add_argument('--plot_data', help='path to test data, generated by caffe parse_log.py')
    parser.add_argument('--output_png_path', help='path to png, will be overwritten by this script')
    # parser.set_defaults(DEBUG=True)
    args = parser.parse_args()

    ref_data = load_data(os.path.join(os.path.dirname(__file__), args.reference_data))
    plot_data = load_data(args.plot_data)
    # cut ref data to plot data
    plot_x_len = ref_data.__len__()
    if ref_data.__len__() > plot_data.__len__():
        plot_x_len = plot_data.__len__()
    print 'found {} viable datapoints to plot'.format(plot_x_len)
    # 0:Iters 1:Seconds 2:TestAccuracy 3:TestLoss
    test_iterations = [x[0] for x in plot_data[:plot_x_len]]
    test_accuracy = [x[2] for x in plot_data[:plot_x_len]]
    test_loss = [x[3] for x in plot_data[:plot_x_len]]
    ref_iterations = [x[0] for x in ref_data[:plot_x_len]]
    ref_accuracy = [x[2] for x in ref_data[:plot_x_len]]
    ref_loss = [x[3] for x in ref_data[:plot_x_len]]

    plt.ioff()

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    line1a, = ax1.plot(test_iterations, test_loss, label="Test Loss", color='r')
    line1b, = ax1.plot(ref_iterations, ref_loss, label="Referenz loss", color='#eb977f')
    line2a, = ax2.plot(test_iterations, test_accuracy, label="Test Accuracy", color='b')
    line2b, = ax2.plot(ref_iterations, ref_accuracy, label="Referenz Accuracy", color='#99cbe9')
    ax1.set_xlabel('Iterationen')
    ax1.set_ylabel('Loss', color='r')
    ax2.set_ylabel("Accuracy", color='b')

    legend = plt.legend(handles=[line1a, line1b, line2a, line2b], bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    print 'saving plot png to ' + args.output_png_path
    pylab.savefig(args.output_png_path, bbox_extra_artists=(legend,), bbox_inches='tight')
