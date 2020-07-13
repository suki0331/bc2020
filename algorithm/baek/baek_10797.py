import sys
y = int(input())
a,b,c,d,e = map(int, sys.stdin.readline().split())
f = (a,b,c,d,e)
count = 0
for x in f:
    if x == y:
        count += 1

print(count)