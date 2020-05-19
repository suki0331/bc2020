# 2. tuple
# 리스트와 거의 같으나 삭제 및 수정이 안된다.
a = (1, 2, 3)
b = 1, 2, 3 
print(type(a))
print(type(b))

# a.remove(2) # compile error
print(a)
print(a + b)
print(a * 3)

# print(a - 3) # cannot remove or overwrite
