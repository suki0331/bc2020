# escape from a rectangle
import sys
a,b,c,d = map(int, sys.stdin.readline().split()) 

# print the shortest 
# route min(route)
print(min((c-a),(d-b),a,b))