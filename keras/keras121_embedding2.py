# from keras.preprocessing.text import Tokenizer
# import numpy as np

# docs = ["너무 재밌어요", "참 최고에요", "참 잘 만든 영화에요", '추천하고 싶은 영화입니다', '한번 더 보고 싶네요', 
#         '글쎄요', '별로에요', '생각보다 지루해요', '연기가 어색해요', '재미없어요', '너무 재미없다', '참 재밌네요']

# # 긍정1, 부정0
# labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1])

# # 토큰화
# token = Tokenizer()
# token.fit_on_texts(docs)
# print("token.word_index: \n", token.word_index)
# # token.word_index:
# #  {'너무': 1, '참': 2, '재밌어요': 3, '최고에요': 4, '잘': 5, '만든': 6, '영화에요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10, '한번': 11, '더': 12, '보고': 13, '싶네요': 14, '글쎄요': 15, '별로에
# # 요': 16, '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20, '재미없어요': 21, '재미없다': 22, '재밌네요': 23}
# # 중복 제거된 인덱싱. 


# # '참'이라는 단어를 3번 주면?
# # 많이 사용하는 단어가 인덱싱 우선순위가 됨
# # token.word_index:
# #  {'참': 1, '너무': 2, '재밌어요': 3, '최고에요': 4, '잘': 5, '만든': 6, '영화에요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10, '한번': 11, '더': 12, '보고': 13, '싶네요': 14, '글쎄요': 15, '별로에
# # 요': 16, '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20, '재미없어요': 21, '재미없다': 22, '재밌네요': 23}

# x = token.texts_to_sequences(docs)
# print("token.texts_to_sequences: \n", x)
# # 문자를 수치화
# # token.texts_to_sequences:
# #  [[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14], [15], [16], [17, 18], [19, 20], [21], 
# # [2, 22], [1, 23]]

# # 문제점? shape가 동일 하지 않는 점
# # shape를 맞춰줘야함. 하나하나를 reshape 할 수 없음
# # pad_sequences를 써준다면! padding을 쓰면 빈자리에 0을 넣어서 진행
# # 제일 큰 shape의 숫자를 맞춰서 나머지는 0으로 채우면 동일한 shape로 됨
# # LSTM의 경우 : 의미 있는 인덱싱이 뒤로 가는 것이 좋을 수 있음


# from keras.preprocessing.sequence import pad_sequences

# pad_x = pad_sequences(x, padding='pre')
# print("pad_sequences_pre: \n", pad_x)
# # pad_sequences_pre:
# #  [[ 0  0  2  3]
# #  [ 0  0  1  4]
# #  [ 1  5  6  7]
# #  [ 0  8  9 10]
# #  [11 12 13 14]
# #  [ 0  0  0 15]
# #  [ 0  0  0 16]
# #  [ 0  0 17 18]
# #  [ 0  0 19 20]
# #  [ 0  0  0 21]
# #  [ 0  0  2 22]
# #  [ 0  0  1 23]]
# # padding='pre' 앞에서부터 0을 채워준다.

# pad_x = pad_sequences(x, padding='post')
# print("pad_sequences_post: \n", pad_x)
# # pad_sequences_post:
# #  [[ 2  3  0  0]
# #  [ 1  4  0  0]
# #  [ 1  5  6  7]
# #  [ 8  9 10  0]
# #  [11 12 13 14]
# #  [15  0  0  0]
# #  [16  0  0  0]
# #  [17 18  0  0]
# #  [19 20  0  0]
# #  [21  0  0  0]
# #  [ 2 22  0  0]
# #  [ 1 23  0  0]]
# # padding='post' 뒤에서부터 0을 채워준다.

# pad_x = pad_sequences(x, value=1.0)
# print("pad_sequences_value: \n", pad_x)
# # pad_sequences_value:
# #  [[ 1  1  2  3]
# #  [ 1  1  1  4]
# #  [ 1  5  6  7]
# #  [ 1  8  9 10]
# #  [11 12 13 14]
# #  [ 1  1  1 15]
# #  [ 1  1  1 16]
# #  [ 1  1 17 18]
# #  [ 1  1 19 20]
# #  [ 1  1  1 21]
# #  [ 1  1  2 22]
# #  [ 1  1  1 23]]
# # value=1.0는 0이 아닌 value 값으로 채워짐


# ''' 명석이 소스
# from keras.preprocessing.text import Tokenizer
# import numpy as np

# docs = ["너무 재밋어요", "최고에요", "참 잘 만든 영화에요",
#         '추천하고 싶은 영화입니다', '한번 더 보고 싶네요', '글쎄요',
#         '별로에요', '생각보다 지루해요', '연기가 어색해요', 
#         '재미없어요', '너무 재미없다', '참 재밋네요']

# # 긍정 1, 부정 0

# labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1])


# # 토큰화
# token = Tokenizer()
# token.fit_on_texts(docs)
# # print(token.word_index) # 중복 된 단어들은 앞 쪽으로 몰림 그리고 한번만 등장(인덱스 번호니까) => 많이 등장하는 놈이 맨 앞으로

# x = token.texts_to_sequences(docs)
# # print(x)

# from keras.preprocessing.sequence import pad_sequences
# # 패드 시퀀스 
# '''
# '''
# (2,) [3,7]
# (1,) [2]
# (3,) [4,5,11]
# (5,) [5,4,3,2,6]
# '''
# '''

# pad_x_pre = pad_sequences(x, padding = 'pre')
# print(pad_x_pre)

# pad_x = pad_sequences(x, padding = 'post',value = 1.0)
# print(pad_x)
# '''

#판서
# 공주 1|

# 왕자 2|

# 빵   3|

# 밥   4|

# 돈   5|

# 원핫인코딩을 압축하는게 embedding





# 모델 적용!
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
# print(token.word_index) # 중복 된 단어들은 앞 쪽으로 몰림 그리고 한번만 등장(인덱스 번호니까) => 많이 등장하는 놈이 맨 앞으로

x = token.texts_to_sequences(docs)
# print(x)

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
model.add(Embedding(word_size, 10, input_length=5))
# model.add(Embedding(25, 10, input_length=5))  # embedding parameter = (none, 5, 10)
# model.add(Embedding(25, 10))    #안돌아감
model.add(LSTM(10))
# enbedding = 다음에 댄스나 lstm 엮기 전에 와꾸 맞추는중 

# word_size = 전체 단어에 갯수 25
# 두번쨰 10은 아웃풋!(출력 노드에 갯수: 10000000을 서도 문제가 없다.)  ( 다른 모델들은 앞에 아웃풋이 들어가지만 embedding 은 두번쨰 들어간다.)
# input_length는 (12, 5) 에서 5임

# 최종출력은 긍정인가 부정인가를 판단해서 1, 0 으로 나와야됌
# model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.summary()
'''
embedding 의 서머리 파라미터 계산은?

_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding_1 (Embedding)      (None, 5, 10)             250
_________________________________________________________________
flatten_1 (Flatten)          (None, 50)                0
_________________________________________________________________
dense_1 (Dense)              (None, 2)                 102
=================================================================
Total params: 352
Trainable params: 352
Non-trainable params: 0
_________________________________________________________________
'''

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(pad_x, labels, epochs=30)

acc = model.evaluate(pad_x, labels)[1]  # ([1]acc 값 뽑겠다라는 뜻)
print("acc : ", acc)


# embedding에 word_size를 25에서 250으로 바꿨는데도 모델이 돌아가는 이유는?
# 파라미터 연산 갯수와 영향이 있음 