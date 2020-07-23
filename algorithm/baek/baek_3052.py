x1 = int(input())
x2 = int(input())
x3 = int(input())
x4 = int(input())
x5 = int(input())
x6 = int(input())
x7 = int(input())
x8 = int(input())
x9 = int(input())
x10 = int(input())

e = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]
a = []

for number in e:
    n = number%42  
    if n not in a:
        a.append(n)
print(len(a))