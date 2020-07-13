home_to_school = int(input())
school_to_pc = int(input())
pc_to_academy = int(input())
academy_to_home = int(input())

elapsed_time = home_to_school + school_to_pc + pc_to_academy + academy_to_home

min = elapsed_time//60

sec = elapsed_time%60

print(min)
print(sec)