#!/usr/bin/python
# -*- coding: utf-8 -*-

import calendar
import logging
import exifutil
import os
import cStringIO as StringIO
from flask import Flask, jsonify, render_template, request
from werkzeug import secure_filename
from flask.json import JSONEncoder
from flask_compress import Compress
from classifier import NNClassifier
from datetime import datetime
from PIL import Image

log = logging.getLogger(__name__)
compress = Compress()


class NNWebInterface(Flask):
    def __init__(self, import_name, classifiers, upload_folder, **kwargs):
        super(NNWebInterface, self).__init__(import_name)
        compress.init_app(self)
        self.route("/", methods=['GET'])(self.index)
        self.route("/classify_upload", methods=['POST'])(self.classify_upload)
        self.route("/classify_url", methods=['GET'])(self.classify_url)
        self.route("/generate_json", methods=['GET'])(self.generate_json)
        self.classifiers_json = classifiers
        self.flask_upload_folder = upload_folder

    def classify_upload(self):
        try:
            # We will save the file to disk for possible data collection.
            imagefile = request.files['imagefile']
            filename_ = datetime.now().strftime('%Y-%m-%d_%Hh-%Mm-%Ss') + \
                        secure_filename(imagefile.filename)
            filename = os.path.join(self.flask_upload_folder, filename_)
            imagefile.save(filename)
            logging.info('Saving to %s.', filename)
            image = exifutil.open_oriented_im(filename)

        except Exception as err:
            logging.info('Uploaded image open error: %s', err)
            return render_template(
                'index.html', has_result=True,
                result=(False, 'Cannot open uploaded image.')
            )

        # start all classifiers
        running_threads = []
        for cnn_index in self.classifiers_json:
            classifier = NNClassifier(self.classifiers_json[cnn_index], gpu_mode=False)
            if classifier.set_image(image):
                classifier.start()
                running_threads.append(classifier)
            else:
                log.error('something wrong with classifier: {}'.format(self.classifiers_json[cnn_index]['name']))

        # gather results
        results = []
        tag_list = []
        for classifier in running_threads:
            classifier.join()
            success, result = classifier.get_result()
            if not success:
                log.error('something wrong with classifier: {}\nresult: {}'.format(classifier.get_name(), result))
                continue
            results.append(result)
            for category_score in result[1]:
                for tag in category_score[0].split(', '):
                    tag_list.append(tag)
        return render_template(
            'index.html', has_result=True, results=results,
            imagesrc=self.embed_image_html(image), tag_list=tag_list
        )

    def classify_url(self):
        image_url = request.args.get('imageurl', '')
        logging.info('Image: %s', image_url)

        # start all classifiers
        running_threads = []
        for cnn_index in self.classifiers_json:
            classifier = NNClassifier(self.classifiers_json[cnn_index], gpu_mode=False)
            if classifier.set_image_url(image_url):
                classifier.start()
                running_threads.append(classifier)
            else:
                log.error('something wrong with classifier: {}'.format(self.classifiers_json[cnn_index]['name']))

        # gather results
        results = []
        tag_list = []
        for classifier in running_threads:
            classifier.join()
            success, result = classifier.get_result()
            if not success:
                log.error('something wrong with classifier: {}\nresult: {}'.format(classifier.get_name(), result))
                continue
            results.append(result)
            for category_score in result[1]:
                for tag in category_score[0].split(', '):
                    tag_list.append(tag)
        return render_template(
            'index.html', has_result=True, results=results,
            imagesrc=image_url, tag_list=tag_list
        )

    @staticmethod
    def generate_json():
        tag_list = request.args.get('json_data')
        json_data = {"tags": tag_list}
        return jsonify(**json_data)

    @staticmethod
    def index():
        logging.debug('index accessed')
        return render_template('index.html', has_result=False)

    @staticmethod
    def embed_image_html(image):
        """Creates an image embedded in HTML base64 format."""
        image_pil = Image.fromarray((255 * image).astype('uint8'))
        image_pil = image_pil.resize((256, 256))
        string_buf = StringIO.StringIO()
        image_pil.save(string_buf, format='png')
        data = string_buf.getvalue().encode('base64').replace('\n', '')
        return 'data:image/png;base64,' + data

