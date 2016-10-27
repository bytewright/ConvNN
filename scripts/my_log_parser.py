import os
import sys
import configargparse


if __name__ == "__main__":
    configpath = os.path.join(os.path.dirname(__file__), 'parser_config.ini')
    parser = configargparse.ArgParser(default_config_files=[configpath])
    parser.add_argument('--caffe_log_path', help='file of captured stdout and stderr')
    parser.add_argument('--output_test_path', help='file of captured stdout and stderr')
    parser.add_argument('--output_train_path', help='file of captured stdout and stderr')
    # parser.set_defaults(DEBUG=True)
    args = parser.parse_args()

