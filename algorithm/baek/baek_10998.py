# https://www.acmicpc.net/problem/10998
# 두 정수 A와 B를 입력받은 다음, A×B를 출력하는 프로그램을 작성하시오.

def multiply():
    a, b = map(int, input().split())
    c = a*b
    print(c)

if __name__ == '__main__':
    multiply()