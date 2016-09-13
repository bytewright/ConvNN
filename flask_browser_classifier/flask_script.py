import os
from flask import Flask, request, redirect, url_for, send_from_directory, jsonify
from werkzeug import secure_filename
import subprocess

#http://flask.pocoo.org/docs/0.11/patterns/fileuploads/#uploading-files
#UPLOAD_FOLDER = 'static'
UPLOAD_FOLDER = '/home/ellerch/caffeProject/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__, static_folder='/home/ellerch/caffeProject/images')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file'] # [0]
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
        else:
            return 'error'
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    proc = subprocess.Popen(['python', '/home/ellerch/caffeProject/bvlc_alexnet_test/my_script.py',  filename], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = proc.communicate()[0]
    #output = output.split('Network initialization done.')[1]
    if('\n{{' in output):
        output = '{{' + output.split('\n{{')[1]
        #output = output.replace('[IMG_SRC]', '/upload/'+filename)
        output = output[:len(output)-1]
        output = output.replace('},{', '},<br>{')
    #return jsonify(output)
    return output
    
if __name__ == '__main__':
    app.run()
