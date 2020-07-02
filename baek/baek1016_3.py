# print (prime numbers)^2  < sqrt(max)
import math
n=1000
m = int(math.sqrt(n))
a = [False,False] + [True]*(m-1)
primes=[]

for i in range(2,m+1):
  if a[i]:
    primes.append(i*i)
    for j in range(2*i, m+1, i):
        a[j] = False
print(primes)
