min, max = map(int, input().split())

b = [False,False] + [True]*(max+1)
primes=[]

for i in range(2,max+1):
  if b[i]:
    primes.append(i)
    for j in range(2*i, max+1, i):
        b[j] = False
print(primes)
