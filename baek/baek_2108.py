import sys

N = int(sys.stdin.readline())
a = list(range(N))
for i in range(N):
    a[i] = int(sys.stdin.readline())
a.sort()

# Average
avg = round(sum(a)/N)
# print(avg)

# median
median = a[(N-1)//2]
print(median)

# range
range_ = int(abs(a[-1]-a[0]))
# print(range_)

# mode
for i in range(a):




print(avg)
print(median)
print(mode)
print(range_)