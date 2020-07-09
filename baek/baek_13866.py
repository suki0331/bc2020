# 두 팀의 스킬 레벨 차이의 최솟값을 출력한다.
# 4명의 스킬 레벨이 각각 주어짐
# 2명씩 두 팀을 구성, 두 팀의 스킬레벨의 차가 최소가 될 때 출력
import sys

a,b,c,d = map(int, sys.stdin.readline().split())
e = [a,b,c,d]
e.sort()
print(abs(e[0]-e[1]-e[2]+e[3]))

