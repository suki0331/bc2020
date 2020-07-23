p=[0.2, 0.2, 0.2, 0.2, 0.2]
world=['green', 'red', 'red', 'green', 'green']
measurements = ['red', 'green']
pHit = 0.6
pMiss = 0.2


# for i in range(len(p)-len(measurements)+1):
#     a = [world[i+j] for j in range(len(measurements))]
#     print(a)
def sense(p, Z):
    q=[]
    for i in range(len(p)):
        hit = (Z == world[i])
        q.append(p[i] * (hit * pHit + (1-hit) * pMiss))
    s = sum(q)
    for i in range(len(q)):
        q[i] = q[i] / s
    return q

def pd(q, measurements):
    for i in range(len(p)-len(measurements)+1):
        if ([world[i+j] for j in range(len(measurements))] == measurements):
            q[i] = q[i]*2
    s = sum(q)
    for i in range(len(q)):
        q[i] = q[i] / s
    return q
q = sense(p, measurements)
k = pd(q, measurements)
print(k)
