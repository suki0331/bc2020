a = [[b for b in input()] for c in range(8)]

d = 0
for i in range(8):
    for j in range(8):
        if (i+j)%2 == 0 and a[i][j]=="F":
            d += 1
print(d)