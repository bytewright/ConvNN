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
from datetime import datetime
from PIL import Image

log = logging.getLogger(__name__)
compress = Compress()


class NNWebInterface(Flask):
    def __init__(self, import_name, classifiers, upload_folder, **kwargs):
        super(NNWebInterface, self).__init__(import_name)
        compress.init_app(self)
        self.route("/", methods=['GET'])(self.index)
        #self.route("/set_classifier", methods=['GET', 'POST'])(self.set_classifier)
        self.route("/classify_upload", methods=['POST'])(self.classify_upload)
        self.route("/classify_url", methods=['GET'])(self.classify_url)
        self.classifiers = classifiers
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

        # gather results from all classifiers
        results = []
        for classifier in self.classifiers:
            result = classifier.dummy_classify(image)
            #result = classifier.classify_image(image)
            results.append(result)
        return render_template(
            'index.html', has_result=True, result=results[0],
            imagesrc=self.embed_image_html(image)
        )

    def classify_url(self):
        image_url = request.args.get('imageurl', '')
        logging.info('Image: %s', image_url)

        results = []
        try:
            for classifier in self.classifiers:
                result = classifier.classify_url(image_url)
                results.append(result)
        except Exception as err:
            logging.info('URL error: %s', err)
            return render_template(
                'index.html', has_result=True,
                result=(False, 'Cannot open image url.')
            )

        return render_template(
            'index.html', has_result=True, result=results[0], imagesrc=image_url)


    @staticmethod
    def index():
        log.debug('index accessed')
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

