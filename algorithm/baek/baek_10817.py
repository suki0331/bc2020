import sys
A, B, C = map(int, sys.stdin.readline().split())
D = [A,B,C]
D.sort()
print(D[1])

