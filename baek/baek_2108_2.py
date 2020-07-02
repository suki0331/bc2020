a=[1,1,2,2,2,3,4,5,6,7,7,7,8,9,10]
temp = []

for i in a:
    temp.append(a.count(i))
print(temp)


for i in a:
    if a.count(i) == max(temp):
        