import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer   # tokenizer API

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!'  # tokenizer automatically distinguishes exclamation mark
]

# create an instance of a tokenizer object
tokenizer = Tokenizer(num_words = 100)  # num_words parameter == maximum number of words to keep
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)