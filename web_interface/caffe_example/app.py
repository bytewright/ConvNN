import cPickle
import cStringIO as StringIO
import datetime
import logging
import os
import time
import urllib

import configargparse
import flask
import numpy as np
import pandas as pd
import tornado.httpserver
import tornado.wsgi
import werkzeug
from PIL import Image

import exifutil

#import caffe

#REPO_DIRNAME = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../..')
UPLOAD_FOLDER = '/tmp/caffe_demos_uploads'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])

# Obtain the flask app object
app = flask.Flask(__name__)


@app.route('/')
def index():
    return flask.render_template('index.html', has_result=False)


@app.route('/classify_url', methods=['GET'])
def classify_url():
    imageurl = flask.request.args.get('imageurl', '')
    try:
        string_buffer = StringIO.StringIO(
            urllib.urlopen(imageurl).read())
        #image = caffe.io.load_image(string_buffer)

    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('URL Image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open image from URL.')
        )

    logging.info('Image: %s', imageurl)
    #result = app.clf.classify_image(image)
    #return flask.render_template(
    #    'index.html', has_result=True, result=result, imagesrc=imageurl)
    return flask.render_template('index.html', has_result=True, imagesrc=imageurl)


@app.route('/classify_upload', methods=['POST'])
def classify_upload():
    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(args.flask_upload_folder, filename_)
        imagefile.save(filename)
        logging.info('Saving to %s.', filename)
        image = exifutil.open_oriented_im(filename)

    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')
        )

    result = app.clf.classify_image(image)
    return flask.render_template(
        'index.html', has_result=True, result=result,
        imagesrc=embed_image_html(image)
    )


def embed_image_html(image):
    """Creates an image embedded in HTML base64 format."""
    image_pil = Image.fromarray((255 * image).astype('uint8'))
    image_pil = image_pil.resize((256, 256))
    string_buf = StringIO.StringIO()
    image_pil.save(string_buf, format='png')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/png;base64,' + data


def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS
    )


class ImageClassifier:
    #default_args = {
    #    'model_def_file': (
    #        '{}/models/bvlc_reference_caffenet/deploy.prototxt'.format(REPO_DIRNAME)),
    #    'pretrained_model_file': (
    #        '{}/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'.format(REPO_DIRNAME)),
    #    'mean_file': (
    #        '{}/python/caffe/imagenet/ilsvrc_2012_mean.npy'.format(REPO_DIRNAME)),
    #    'class_labels_file': (
    #        '{}/data/ilsvrc12/synset_words.txt'.format(REPO_DIRNAME)),
    #    'bet_file': (
    #        '{}/data/ilsvrc12/imagenet.bet.pickle'.format(REPO_DIRNAME)),
    #}
    #for key, val in default_args.iteritems():
    #    if not os.path.exists(val):
    #        raise Exception(
    #            "File for {} is missing. Should be at: {}".format(key, val))
    #default_args['image_dim'] = 256
    #default_args['raw_scale'] = 255.
    model_args = {
        'model_path_prefix': '',
        'model_definition': '',
        'model_weights': '',
        'db_mean_file': '',
        'db_labels': '',
        'bet_file': '',
        'image_dim': (100, 100),
        'image_raw_scale': 99
    }
    #todo set dims to correct val for placesdb

    def __init__(self, gpu_mode):
        if gpu_mode:
            logging.info("using GPU mode")
            #caffe.set_mode_gpu()
        else:
            logging.info("using CPU mode")
            #caffe.set_mode_cpu()
        #self.load_classifier()

    def load_classifier(self):
        logging.info('Loading net and associated files...')
        #self.net = caffe.Classifier(
        #    self.model_args['model_definition'],self.model_args['model_weights'],
        #    image_dims=(self.model_args['image_dim']), raw_scale=self.model_args['image_raw_scale'],
        #    mean=np.load(self.model_args['db_mean_file']).mean(1).mean(1), channel_swap=(2, 1, 0)
        #)
        with open(self.model_args['db_labels']) as f:
            labels_df = pd.DataFrame([
                                         {
                                             # todo an places labels anpassen
                                             'synset_id': l.strip().split(' ')[0],
                                             'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
                                         }
                                         for l in f.readlines()
                                         ])
        self.labels = labels_df.sort('synset_id')['name'].values
        self.bet = cPickle.load(open(self.model_args['bet_file']))
        # A bias to prefer children nodes in single-chain paths
        # I am setting the value to 0.1 as a quick, simple model.
        # We could use better psychological models here...
        self.bet['infogain'] -= np.array(self.bet['preferences']) * 0.1

    def classify_image(self, image):
        try:
            starttime = time.time()
            scores = self.net.predict([image], oversample=True).flatten()
            endtime = time.time()
            logging.debug(scores)
            indices = (-scores).argsort()[:5]
            predictions = self.labels[indices]

            # In addition to the prediction text, we will also produce
            # the length for the progress bar visualization.
            meta = [
                (p, '%.5f' % scores[i])
                for i, p in zip(indices, predictions)
            ]
            logging.info('result: %s', str(meta))

            # Compute expected information gain
            expected_infogain = np.dot(
                self.bet['probmat'], scores[self.bet['idmapping']])
            expected_infogain *= self.bet['infogain']

            # sort the scores
            infogain_sort = expected_infogain.argsort()[::-1]
            bet_result = [(self.bet['words'][v], '%.5f' % expected_infogain[v])
                          for v in infogain_sort[:5]]
            logging.info('bet result: %s', str(bet_result))

            return (True, meta, bet_result, '%.3f' % (endtime - starttime))

        except Exception as err:
            logging.info('Classification error: %s', err)
            return (False, 'Something went wrong when classifying the '
                           'image. Maybe try another one?')

    def set_model_args(self,
                       new_path=model_args['model_path_prefix'],
                       new_model_def=model_args['model_definition'],
                       new_weights=model_args['model_weights'],
                       new_mean_file=model_args['db_mean_file'],
                       new_labels=model_args['db_labels'],
                       new_bet_file=model_args['bet_file']):
        self.model_args['model_path_prefix'] = new_path
        self.model_args['model_definition'] = new_model_def
        self.model_args['model_weights'] = new_weights
        self.model_args['db_mean_file'] = new_mean_file
        self.model_args['db_labels'] = new_labels
        self.model_args['bet_file'] = new_bet_file
        logging.info('setting arguments:\n{}: {}\n{}: {}\n{}: {}\n{}: {}\n{}: {}\n{}: {}\n'.format(
            'model_path_prefix', new_path,
            'model_definition', new_model_def,
            'model_weights', new_weights,
            'db_mean_file', new_mean_file,
            'db_labels', new_labels,
            'bet_file', new_bet_file
            ))
        self.load_classifier()


def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def get_args():
    configpath = os.path.join(os.path.dirname(__file__), 'config.ini')
    parser = configargparse.ArgParser(default_config_files=[configpath])
    parser.add_argument(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_argument(
        '-p', '--port',
        help="which port to serve content on",
        type=int, default=5000)
    parser.add_argument(
        '-g', '--gpu',
        help="use gpu mode",
        action='store_true', default=False)
    parser.add_argument('--flask-upload-folder', type=str,
                        help='text file, each line should be one path to a solver file')
    parser.add_argument('--path-prefix', type=str,
                        help='text file, each line should be one path to a solver file')
    parser.add_argument('--model-def-file', type=str,
                        help='text file, each line should be one path to a solver file')
    parser.add_argument('--pretrained-model-file', type=str,
                        help='text file, each line should be one path to a solver file')
    parser.add_argument('--mean-file', type=str,
                        help='text file, each line should be one path to a solver file')
    parser.add_argument('--labels-file', type=str,
                        help='text file, each line should be one path to a solver file')
    return parser.parse_args()


def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """


args = get_args()
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    logFormatter = logging.Formatter("%(asctime)s [%(module)14s] [%(levelname)5s] %(message)s")
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logging.getLogger().addHandler(consoleHandler)
    #start_from_terminal(app)
    # log = logging.getLogger()
    print('{}:{}'.format(args.model_def_file, os.path.exists(args.model_def_file)))
    print('{}:{}'.format(args.pretrained_model_file, os.path.exists(args.model_def_file)))
    print('{}:{}'.format(args.mean_file, os.path.exists(args.model_def_file)))
    print('{}:{}'.format(args.labels_file, os.path.exists(args.model_def_file)))
    print('port:{}'.format(args.port))

    classifier = ImageClassifier(True)
    classifier.set_model_args(args.path_prefix,
                              args.model_def_file,
                              args.pretrained_model_file,
                              args.mean_file,
                              args.labels_file)
    #logging.info('starting with Parameters:\n{}\n{}\n{}\n{}\n{}\n{}'.format(
    #    classifier.model_args['model_path_prefix'],
    #    classifier.model_args['model_definition'],
    #    classifier.model_args['model_weights'],
    #    classifier.model_args['db_mean_file'],
    #    classifier.model_args['db_labels'],
    #    classifier.model_args['bet_file']))

    if not os.path.exists(args.flask_upload_folder):
        logging.info('creating upload folder at:\n{}'.format(args.flask_upload_folder))
        os.makedirs(args.flask_upload_folder)
    # Initialize classifier + warm start by forward for allocation
    app.clf = classifier
    app.clf.net.forward()

    if args.debug:
        app.run(debug=True, host='0.0.0.0', port=args.port)
    else:
        start_tornado(app, args.port)
