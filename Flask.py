import os
from flask import Flask, render_template, request, redirect, url_for
import pickle
import tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
import skimage
from skimage.io import imread

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/upload', methods=['POST','GET'])
def upload_file():

    file = request.form['image']


    text = file
    dic = {0: 'Depressed', 1: 'Not Depressed'}

    # loading the tockenizer

    k = tokenizer.texts_to_sequences([text])
    k = pad_sequences(k, maxlen=100)
    result= dic[np.argmax(model.predict(k))]
    return render_template('index.html',result = result)






if __name__ == '__main__':
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    # Load the model
    model = load_model('model.h5')

    app.run(debug=True)