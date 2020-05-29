a = { 'name' : 'yun',
      'phone' : '010',
      'birth' : '0511'
}

# for (i = 1, i = 100, i++){
#     a = i + a
# }
# print(a)

# for i in 100:

for i in a.keys():
    print(i)
    print()

for values in a.values():
    print(values)
    print()

for keys,values in a.items():
    print(keys,values)
    print()
    
for items in a.items():
    print(items)
    print()

for keys in a.keys():
    print(a.keys())
    print()

for items in a.items():
    print(a.items())
    print()

a = [1,2,3,4,5,6,7,8,9,11]

for i in a:
    i = i*i
    print(i)
    print('-------------')
print("end")

for i in a:
    print(i)

## while문
'''
while 조건문 : # True일 경우 계속 loop
    수행할 문장
'''

## if문
if 1:
    print('True')
else:
    print('False')

if 3:
    print('True')
else:
    print('False')

if 0:
    print('True')
else:
    print('False')

if -1:
    print('True')
else:
    print('False')

# 비교연산자
# <, >, ==, !=, >=, <=

a = 1
if a == 1:
    print("good")

money = 10000
if money >= 30000:
    print('한우먹자')
else:
    print('라면먹자')

### 조건연산자
# and, or, not
money = 20000
card = 1
if money >= 30000 or card == 1:
    print("한우먹자")
else:
    print('라면먹자')

print("===============break==============")
score = [90, 25, 67, 45, 80]
number = 0
for i in score:  
    if i < 30:
        break # 가장 가까운 반복문 중지
    if i >= 60:
        print("합격")
        number += 1

print(f"합격인원 : {number} 명")

print("==============continue===============")
score = [90, 25, 67, 45, 80]
number = 0
for i in score:  
    if i < 60:
        continue 
    if i >= 60:
        print("합격")
        number += 1

print(f"합격인원 : {number} 명")