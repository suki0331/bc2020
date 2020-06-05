import sys
from collections import Counter

# 출력하고 싶은 단어의 수를 첫번째 인자로 입력
try:
    num_words = int(sys.argv[1])
except:
    print("usage: most_common_words.py num_words")
    sys.exit(1) # exit 코드 뒤에 0 외의 숫자가 들어오면 에러를 의미

counter = Counter(word.lower()                      # 소문자로 변환
                  for line in sys.stdin
                  for word in line.strip().split()  # 띄어쓰기 기준으로 나누기
                  if word)                          # 비어 있는 word는 무시

for word, count in counter.most_common(num_words):
    sys.stdout.write(str(count))
    sys.stdout.write("\t")
    sys.stdout.write(word)
    sys.stdout.write("\n")