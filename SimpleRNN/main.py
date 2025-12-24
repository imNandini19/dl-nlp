#1. import all libraries
import numpy as np
import tensorflow as tf
import re
import streamlit as st
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model

#load dataset
word_index=imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}

# load pre-trained model
model = load_model('SimpleRNN/simple_rnn_imdb.h5')

# 2.helper function
# clean the input
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove punctuation
    return text
# function to decode the review
def preprocess_text(text):
    text = clean_text(text)
    words = text.split()
    encoded_review = [word_index.get(word, 2) for word in words]  # 2 = OOV
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

## 3. prediction function
def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]
    

st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positivie or negative')

#user input
user_input=st.text_area('Movie Review')

if st.button('Classify'):
    preprocessed_input=preprocess_text(user_input)
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0] >= 0.5 else 'Negative'
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review.')

