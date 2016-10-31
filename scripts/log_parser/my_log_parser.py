import csv
import configargparse


if __name__ == "__main__":
    parser = configargparse.ArgParser()
    parser.add_argument('--log_file', help='path to test data')
    parser.add_argument('--output_file', help='path to output csv, will be overwritten by this script')
    args = parser.parse_args()

    lines = [['iter', 'acc', 'loss']]
    start_search = False
    got_iteration = False
    current_row = []
    with open(args.log_file, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if not start_search and 'Network initialization done.' in line:
                start_search = True
            if start_search:
                if got_iteration and 'solver.cpp:404]     Test net output #0' in line:
                    acc = line.split('accuracy = ')[1]
                    current_row.append(acc)
                if got_iteration and 'solver.cpp:404]     Test net output #1' in line:
                    loss_test = line.split('loss = ')[1]
                    loss_test = loss_test.split(' ')[0]
                    current_row.append(loss_test)
                    lines.append(current_row)
                    got_iteration = False
                if 'solver.cpp:228] Iteration ' in line:
                    iter, loss_train = line.split(',')
                    iter = iter.split(' Iteration ')[1]
                    current_row = [iter]
                    got_iteration = True
    #lines = interpolate_to_500_step(lines)
    with open(args.output_file, 'wb') as csvfile:
        logwriter = csv.writer(csvfile, delimiter=';')
        for line in lines:
            logwriter.writerow(line)
        print 'wrote {} lines to {}'.format(lines.__len__(), args.output_file)
