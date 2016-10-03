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
    def __init__(self, import_name, classifier, upload_folder, **kwargs):
        super(NNWebInterface, self).__init__(import_name)
        compress.init_app(self)
        self.route("/", methods=['GET'])(self.index)
        self.route("/set_classifier", methods=['GET', 'POST'])(self.set_classifier)
        self.route("/classify_upload", methods=['POST'])(self.classify_upload)
        self.classifier = classifier
        self.flask_upload_folder = upload_folder
        self.classifier_list = []

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

        result = self.classifier.classify_image(image)
        return render_template(
            'index.html', has_result=True, result=result,
            imagesrc=self.embed_image_html(image)
        )

    def set_other_classifiers(self, new_classifiers):
        self.classifier_list = new_classifiers

    def set_classifier(self):
        return self.classifier_list

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

