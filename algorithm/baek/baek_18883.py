# Print 1 to N*M as example when positive integer N,M is given
import sys

N, M = map(int, sys.stdin.readline().split())
K = N*M

# Reordered the code due to conditions
for i in range(K):    
    if (i+1)%M ==0:
        print(i+1)
        continue
    print(i+1, end=' ')