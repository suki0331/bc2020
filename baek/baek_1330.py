# https://www.acmicpc.net/problem/1330
# 두 정수 A와 B가 주어졌을 때, A와 B를 비교하는 프로그램을 작성하시오.

def compare():
    a, b = map(int, input().split())
    if a>b:
        print(">")
    elif a<b:
        print("<")
    elif a==b:
        print("==")

if __name__ == '__main__':
    compare()