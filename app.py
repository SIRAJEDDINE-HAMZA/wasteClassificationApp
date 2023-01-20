from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
import keras

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# path du model de prediction 
MODEL_PATH = 'models/bestmodel.hdf5'

# load model
model = load_model(MODEL_PATH)
print('Model loaded. Check http://127.0.0.1:5000/')

# fonction de prediction "arg => image et le model "
def model_predict(img_path, model):
# image resizing 
    img1 = keras.utils.load_img(img_path, target_size=(155, 155)) 
    img1  = keras.utils.img_to_array(img1) 
    img1 = np.expand_dims(img1, axis=0) 
    result = model.predict(img1/255)
    return result


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
        basepath, 'TEST', secure_filename( f.filename))
        f.save(file_path)
        preds = model_predict(file_path, model)
        acc = preds[0][0]*100
        if   preds[0][1] < 0.5:   
            return f" Objet Organic avec {acc:.2f}% de precision "
        else:
            return f"Objet Recyclable avec {acc:.2f}% de precision"   
    return None


if __name__ == '__main__':
    app.run(debug=True)

