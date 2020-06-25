x = int(input())
n = 0
while True:
    if (n+1)**2 >= x:
        
        
            break
    n += 1

print(n)

while True:
    if x <= n*(n+1):
        ans = 2*n
        break
    else:
        ans = 2*n +1
        break

print(ans)