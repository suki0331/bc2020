import sys
x = int(sys.stdin.readline())
COUNT = 0
y = x
while True:
    COUNT += 1
    a = y//10
    b = y%10
    # print(a)
    # print(b)
    c = (a+b)%10
    # print(c)
    y = 10*b+c
    # print(y)
    if y == x:
        break

print(COUNT)