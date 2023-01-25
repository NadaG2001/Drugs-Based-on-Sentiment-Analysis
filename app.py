from flask import Flask, render_template, url_for, request
import numpy as np
import pandas as pd
import pickle
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
#from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
from numpy import array

app = Flask(__name__)
model = load_model('rnn_model.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    max_length = 200
    if request.method == 'POST':
        review = request.form['review']
        data = [review]
        tokenizer.fit_on_texts(data)
        enc = tokenizer.texts_to_sequences(data)
        enc = pad_sequences(enc, maxlen=max_length, padding='post')
        my_prediction = model.predict(array([enc][0]))[0][0]
        #class1 = model.predict(array([enc][0]))[0][0]
        classes_x = np.round(my_prediction).astype(int)
    return render_template('result.html', prediction=classes_x)


if __name__ == '__main__':
    app.run(debug=True)
