import sys
x1,x2,x3,x4,x5,x6,x7,x8,x9,x10 = map(int, sys.stdin.readline().split())

e = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]
a = []
for number in e:
    n = number%42  
    if number not in a:
        a.append(n)
print(len(a))