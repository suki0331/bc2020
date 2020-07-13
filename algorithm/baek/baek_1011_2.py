import math
T = int(input())
for i in range(T):
    A,B = input().split()
    A = int(A)
    B = int(B)
    len = B-A
    if (len<=3):
        print(len)
    else:
        n = int(math.sqrt(len))
        if( len == n**2):
            print(2*n-1)
        elif(n**2 < len <= n**2+n):
            print(2*n)
        else:
            print(2*n+1)