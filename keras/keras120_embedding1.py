# 자연어 처리의 기본
# 토큰의 개념부터 알아보자
from keras.preprocessing.text import Tokenizer

text = "나는 맛있는 밥을 먹었다"
token = Tokenizer()
token.fit_on_texts([text])

print("token.word_index: \n", token.word_index)
# token.word_index:
#  {'나는': 1, '맛있는': 2, '밥을': 3, '먹었다': 4}

# 문장을 단어 단위로 잘라서 인덱싱(수치화)을 걸어줌
# 띄어쓰기를 3이라 가정하면, 12 3 456 3 78 3 91011

x = token.texts_to_sequences([text])
print("token.texts_to_sequences: \n", x)
# token.texts_to_sequences:
#  [[1, 2, 3, 4]]
# LSTM 모델에 넣고 다음 단어를 예측할 수 있음
# 하지만, 여기서 문제는 2!=1*2, 3!=1*3.. 아니다. 단지 인덱싱일 뿐

# 그럼 원핫인코딩을 해볼까?
from keras.utils import to_categorical

word_size = len(token.word_index) + 1
x = to_categorical(x, num_classes=word_size)

print("to_categorical: \n",x)
# to_categorical:
#  [[[0. 1. 0. 0. 0.]
#   [0. 0. 1. 0. 0.]
#   [0. 0. 0. 1. 0.]
#   [0. 0. 0. 0. 1.]]]
# to_categorical.shape=(4,5)
# 여기서 문제점은? 단어가 늘어날수록 데이터가 방대해지는 문제점

# 그렇다면 해결방법은? 압축해보자. 그래서 나온 개념이 embedding
# embedding 자연어처리, 시계열에서 상당히 많이 사용




''' 명석이 소스
from keras.preprocessing.text import Tokenizer

text = "나는 맛있는 밥을 먹었다."

token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index)
# {'나는': 1, '맛있는': 2, '밥을': 3, '먹었다': 4}

x = token.texts_to_sequences([text])
print(x) # [[1, 2, 3, 4]] 인덱싱된 번호라서 딱히 의미가 있는지는.. 그래서 원핫인코딩 => 그럼 매트릭스가 커짐...


from keras.utils import to_categorical

word_size = len(token.word_index)+1
x = to_categorical(x, num_classes = word_size)
print(x)
'''
'''
1. 토큰(Token) 이란?

  ㅇ [전산 : 프로그래밍 구문]
     - 가장 낮은 단위로 어휘 항목들을 구분할 수 있는 분류 요소
        . 의미를 가지는 최소한의 문자 덩어리(문자열)

2. [프로그래밍 구문]  구문 요소

  ㅇ 가장 낮은 단위로 어휘 항목들을 구분할 수 있는 분류 요소 
     - 예약어(reserved word) : 약속된 문자열
        . 프로그래밍 언어 자체가 사용하는 예약어(키워드)
     - 식별자(identifier)    : 프로그래밍 언어에서 미리 정의되는 언어 구성자
        . 재정의 불가능 식별자 : 예약어
        . 재정의 가능 식별자   : 미리 정의되지만 재정의 가능한 식별자
     - 리터럴(literal,상수)  : 코드 상에 쓰인 값이 실행시 그 값 그대로의 의미를 갖음
        . 수치 리터럴(`132`), 문자 리터럴(`hello`) 등
     - 특수기호              : `;(세미콜론)`,`.(마침표)`,연산 기호(+,-,*,/ 등)

'''
