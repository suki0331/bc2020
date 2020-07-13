T = int(input())
dist = list(range(T))

for i in range(T):
    x, y= map(int, input().split())
    dist[i] = y-x
    n = 0
    while True:
        if (n+1)**2 >= dist[i]:            
            break
        n += 1

    while True:
        if dist[i] <= n*(n+1):
            ans = 2*n            
        else:
            ans = 2*n+1            
        break    
    print(ans)