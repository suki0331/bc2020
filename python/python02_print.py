# print문과 format 함수
a = '사과'
b = '배'
c = '옥수수'

print('선생님은 잘생기셨다.')

print(a)
print(a, b)
print(a, b, c)

print("나는 {0}를 먹었다.".format(a)) # 0 번째 요소인 a 반환
print("나는 {0}와 {1}를 먹었다.".format(a,b)) # 0,1 번째 요소인 a, b 반환
print("나는 {0}와 {1}와 {2}를 먹었다.".format(a,b,c)) # 0, 1, 2 번째 요소인 a, b, c 반환

print('나는', a,'를 먹었다.')
print('나는', a,'와',b,'를 먹었다.')
print('나는', a,'와',b,'와',c,'를 먹었다.')

print('나는 ', a,'를 먹었다.', sep='')
print('나는 ', a,'와 ',b,'를 먹었다.', sep='')
print('나는 ', a,'와 ',b,'와 ',c,'를 먹었다.', sep='')

print('나는', a +'를 먹었다.')
print('나는', a +'와',b +'를 먹었다.')
print('나는', a +'와',b +'와',c +'를 먹었다.')

print('나는 '+ a +'를 먹었다.')
print('나는 '+ a +'와 '+ b +'를 먹었다.')
print('나는 '+ a +'와 '+ b +'와 '+ c +'를 먹었다.')

print(f'나는 {a}와 {b}와 {c}를 먹었다.')