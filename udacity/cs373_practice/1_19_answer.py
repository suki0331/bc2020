'''
#Program a function that returns a new distribution 
#q, shifted to the right by U units. If U=0, q should 
#be the same as p.

p=[0, 1, 0, 0, 0]
world=['green', 'red', 'red', 'green', 'green']
measurements = ['red', 'green']
pHit = 0.6
pMiss = 0.2

def sense(p, Z):
    q=[]
    for i in range(len(p)):
        hit = (Z == world[i])
        q.append(p[i] * (hit * pHit + (1-hit) * pMiss))
    s = sum(q)
    for i in range(len(q)):
        q[i] = q[i] / s
    return q

def move(p, U):
    q = []
    for i in range(len(p)):
        q.append(p[(i-U) % len(p)])
    return q   

print(move(p, 1))

#Alternate solution:

# U = U % len(p)
# q = p[-U:] + p[:-U]

'''
p=[1,2,3,4,5]
print(p[-1])

q = -1
print(q%5)
k = []
for i in range(5):
    k.append(i)

print(k)
