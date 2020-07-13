from collections import Counter
s = input()
print(tuple(s))
count = 0
rm=[]
cp=[]
role_model = [s[i] for i in range(len(s)//2)]
print(f"role_model : {role_model}")
copy = [s[i] for i in range(len(s)-len(s)//2, len(s))]
print(f"copy : {copy}")
copy.reverse()
print(f"reversed_copy : {copy}")

for i in range(len(s)//2):
    if role_model[i] != copy[i]:
        rm.append(role_model[i])
        cp.append(copy[i])
        count += 1
rm = tuple(rm)
cp = tuple(cp) 
# for i in range(len(rm)):
# print(set(rm))
# print(set(rm)==set(cp))
if any(x in rm for x in cp):
    count -= 1
    
print(count)
print(f"count : {count}")

print(f"rm : {rm}")
print(f"cp : {cp}")
