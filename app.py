from flask import Flask, request, render_template
import os
from flask_cors import CORS, cross_origin
from yolov4_image import predict
from werkzeug.utils import secure_filename

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

    

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    f = request.files['file']
    filename = secure_filename(f.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(filepath)
    result = predict(filepath)
    return render_template("uploaded.html", filepath=filepath)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
