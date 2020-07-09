# 어떤 사람의 생년월일과 기준 날짜가 주어졌을 때
# 기준 날짜로 그 사람의 만 나이, 세는 나이, 연 나이를 모두 구하기
import sys

a, b, c = map(int, sys.stdin.readline().split())
x, y, z = map(int, sys.stdin.readline().split())

manyear = x-a
month = y-b
day = z-c
if day<0:
    month -= 1

if month <0:
    manyear -= 1

yeonyear = x-a 


print(manyear)
print(yeonyear+1)
print(yeonyear)