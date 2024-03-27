from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='models/model_8.h5'

# Load your trained model
model = load_model(MODEL_PATH)

def extract_features(filename, model):
    try:
        image = Image.open(filename)

    except:
        print("ERROR: Couldn't open image! Make sure the image path and extension is correct")

    image = image.resize((299,299))
    image = np.array(image)
    # for images that has 4 channels, we convert them into 3 channels
    if image.shape[2] == 4:
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image/127.5
    image = image - 1.0
    feature = model.predict(image)
    return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text


@app.route('/', methods=['GET', 'POST'])
def main():
    # Main page
    return render_template('index.html')



@app.route('/submit', methods=['GET', 'POST'])
def get_output():
    max_length = 32
    tokenizer = load(open("tokenizer.p","rb"))
    model = load_model('models/model_9.h5')
    xception_model = Xception(include_top=False, pooling="avg")

    if request.method == 'POST':
        # Get the file from post request
        f = request.files['my_image']

        # Save the file to ./uploads
        #basepath = os.path.dirname(__file__)
        #file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        #f.save(file_path)

        #photo = extract_features(file_path, xception_model)


        img_path = "static/tests/" + f.filename
        photo = extract_features(img_path, xception_model)
        f.save(img_path)

        #img = Image.open(img_path)
        # Make prediction
        preds = generate_desc(model, tokenizer, photo, max_length)

        return render_template("index.html",  img_path = img_path, prediction = preds[5:-3])
	    #return preds[5:-3]

    return None


if __name__ == '__main__':
    app.run(debug=True)
