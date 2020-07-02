import math
min, max = map(int, input().split())
num = []
a = [True]*(max-min+1)

# print (min,max)
for i in range(0, max-min+1):
    
    if a[i]:
        num.append(min+i)
    

print(num)