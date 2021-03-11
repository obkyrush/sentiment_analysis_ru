import numpy as np
import os
import sys
import pickle
from flask import Flask, request, render_template
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer


vectorizer_filepath = './model/vec_bow.pickle'
model_filepath = './model/logit.pickle'

app = Flask(__name__)
app.vectorizer = pickle.load(open(vectorizer_filepath, 'rb'))
app.model = pickle.load(open(model_filepath, 'rb'))
app.score = 0
app.text = ''

def preprocess_text_ignore_non_letters(text):
    russian_letters = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
    text = text.lower().strip()
    text = ''.join(char for char in text if char in russian_letters or char.isspace())
    text = [word for word in text.split() if word[0] != '@' and word]
    text = ' '.join(text)
    return text

def predict_score(text):
    text_bow = app.vectorizer.transform([text])
    return int(100 * app.model.predict_proba(text_bow)[0, 1])

@app.route('/')
def input_text_form():
    return render_template('input_form.html', score_value=0, default_text=app.text)

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    app.text = text
    processed_text = preprocess_text_ignore_non_letters(text)
    app.score = predict_score(processed_text)
    return render_template('input_form.html', score_value=app.score, default_text=app.text)

app.run()
