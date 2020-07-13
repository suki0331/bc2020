from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences   # padding

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]

tokenizer = Tokenizer(num_words=100, oov_token="<OOV>") # oov= out of vocab
tokenizer.fit_on_texts(sentences)
word_index=tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

padded = pad_sequences(sequences, padding='post', truncating='post', maxlen=5)
print(word_index) 
# {'my': 1, 'love': 2, 'dog': 3, 'i': 4, 'you': 5, 'cat': 6, 'do': 7, 'think': 8, 'is': 9, 'amazing': 10}
# {'<OOV>': 1, 'my': 2, 'love': 3, 'dog': 4, 'i': 5, 'you': 6, 'cat': 7, 'do': 8, 'think': 9, 'is': 10, 'amazing': 11}
print(sequences)
# [[4, 2, 1, 3], [4, 2, 1, 6], [5, 2, 1, 3], [7, 5, 8, 1, 3, 9, 10]]
print(padded)
# after padding
# [[ 0  0  0  5  3  2  4]
#  [ 0  0  0  5  3  2  7]
#  [ 0  0  0  6  3  2  4]
#  [ 8  6  9  2  4 10 11]]

# what if new word exists
test_data = [
    'i really love my dog',
    'my dog loves my manatee'
]
test_seq = tokenizer.texts_to_sequences(test_data)
print(test_seq)
# [[4, 2, 1, 3], [1, 3, 1]]
# it doesn't show words that aren't in word_index in result 
# [[5, 1, 3, 2, 4], [2, 4, 1, 2, 1]] when oov_token="<OOV>"

# for this problem, we can use raggedtensor, but it goes beyond the scope of the lesson
# use simpler solution, padding 

