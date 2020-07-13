# print stars from first row to (2*n-1)th row

A = int(input())

for i in range(A):
    print("*"*(i+1))

for k in range(A-1):
    print("*"*(A-k-1))
