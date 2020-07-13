
# a = input()

# a = ['ab','ca','bc','ab','ca','bc','de','de','de','de','de','de']

# check values front to next, if cond == True, change num['word'] from i to i+1
# check all len(comp)
# print(minimum)

length = 2


s = 'abcabcabcabcdededededede'
s_split = [words for words in s]
print(s_split)
cnt = []
for j in range(1, len(s)//2):
    a = []
    for i in range(0, len(s), j): 
        b = s_split[i]+s_split[i+1]
        a.append(b)
    cnt.append(a)

print(cnt)
# # devide sentence for length = 2
# ans = []
# for i in range(len(sentence)):
#     i





# A = 'hihi'
# B = 'babe'
# C = 2*A+3*B
# print(C)

