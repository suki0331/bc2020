import sys
a = int(sys.stdin.readline())
b = int(sys.stdin.readline())
c = int(sys.stdin.readline())
x = int(sys.stdin.readline())
y = int(sys.stdin.readline())

f = [a,b,c]
g = [x,y]
f.sort()
g.sort()

print(f[0]+g[0]-50)
