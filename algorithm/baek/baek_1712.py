# A = Fixed cost, B = Variable cost, C = Price
# print Break-Even Point, if it doesn't exists, print -1

import sys

A, B, C = map(int, sys.stdin.readline().split())

# condition : d(BEP)/di must not converge to positive value and zero
# when i approches infinity 
if C > B:
    # print BEP
    print(A//(C-B)+1)
else:
    print(-1)

