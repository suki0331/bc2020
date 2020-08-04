import sys

a = int(input())
for i in range(a):
    a1, a2, a3, a4, a5 = map(int, sys.stdin.readline().split())
    b = format(350.34*a1+230.90*a2+190.55*a3+125.30*a4+180.90*a5, "0.2f")
    print("$"+b)