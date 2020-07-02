x, y = map(int, input().split())
a = (x+y)//2
b = x-a
a1 = max(a, b)
b1 = min(a, b)
if a1<0 or b1<0 or x>=1000 or x<0 or y>=1000 or y<0 or (x+y)%2 == 1:
    print(-1)
else:
    print(a1, b1)