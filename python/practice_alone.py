import random

answer = random.randint(1,20)
trial_left = 4
trial_count = 0

question = input(f"기회가 {trial_left}번 남았습니다. 1-20 사이의 숫자를 맞춰보세요 : ")

if question == answer:
    print(f"축하합니다. {trial_count}번만에 숫자를 맞추셨습니다.")
else:
    print()