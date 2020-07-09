from keras.preprocessing.text import Tokenizer
import numpy as np

docs = ["너무 재밋어요", "최고에요", "참 잘 만든 영화에요",
        '추천하고 싶은 영화입니다', '한 번 더 보고 싶네요', '글쎄요',
        '별로에요', '생각보다 지루해요', '연기가 어색해요', 
        '재미없어요', '너무 재미없다', '참 재밋네요']

# 긍정 1, 부정 0

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1])


# 토큰화
token = Tokenizer()
token.fit_on_texts(docs)


x = token.texts_to_sequences(docs)


from keras.preprocessing.sequence import pad_sequences


pad_x_pre = pad_sequences(x, padding = 'pre')
print(pad_x_pre)

pad_x = pad_sequences(x, padding = 'post',value = 1.0)
print(pad_x)            # (12, 5)

word_size = len(token.word_index ) +1
print(" 전체 토큰 사이즈 : ", word_size)        # 25

# 자연어 처리
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, LSTM

model = Sequential()
model.add(Embedding(25, 10)) # input length 표시를 안해주면 conv1d랑 lstm 사용 가능     
model.add(LSTM(3))          # shape 자체가 embedding 을 3차로 해주고 input length를 잡아준다 . 그래서 명시를 안해준다.
# model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding_1 (Embedding)      (None, None, 10)          250         
_________________________________________________________________
lstm_1 (LSTM)                (None, 3)                 168    =     4 * (10(input) + 1(bias) + 3(output)) * 3(output)
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 4
=================================================================
Total params: 422
Trainable params: 422
Non-trainable params: 0
_________________________________________________________________
'''

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(pad_x, labels, epochs=30)

acc = model.evaluate(pad_x, labels)[1]  
print("acc : ", acc)


