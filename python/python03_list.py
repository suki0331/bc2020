# 자료형
# 1. list

a = [1,2,3,4,5]
b = [1,2,3,'a','b']

print(a)
print(b)

print(a[0] + a[3])
print(str(b[0]) + b[3])

print(type(a))
print(a[-2])
print(a[1:3])

a = [1, 2, 3, ['a', 'b', 'c']]
print(a[1])
print(a[-1])
print(a[-1][1]) #b

# 1-2. list slicing
a = [1, 2, 3, 4, 5]
print(a[:2])

# 1-3 add list
a = [1, 2, 3]
b = [4, 5, 6]
print(a + b)

c = [7, 8 ,9, 10]
print(a + c)

print(a * 3)

# print(a[2] + 'hi')
print(str(a[2])+'hi')

f = '5'
# print(a[2]+(f)) // compile ERROR
print(a[2]+int(f))

# 리스트 관련 함수
a = [1, 2, 3]
a.append(4)
print(a)
print()
# a = a.append(5) 
# print(a) # return None

a = [1, 3, 4, 2]
a.reverse()
print(a)

a.sort()
print(a)

a.reverse()
print(a)

print()
print(a.index(3))  # == a[3]
print(a.index(1))  # == a[1]
print()

a.insert(0, 7)
print(a)
a.insert(3,3)
print()

a.remove(7)
print(a)  # == [4,3,3,2,1]

a.remove(3)
print(a) # 가장 앞에있는 3이 삭제됨

a.insert(5,3)
a.remove(3)
print(a)