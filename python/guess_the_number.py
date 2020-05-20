import random

trial_left = 4
trial_count = 1
answer = random.randint(1,20)

while (input!=answer):
    try:
        question = input(f"기회가 {trial_left}번 남았습니다. 1-20 사이의 숫자를 맞춰보세요 : ")
        if int(question) == answer:
            print(f"축하합니다. {trial_count}번만에 숫자를 맞추셨습니다.")
            break
        elif int(question) > answer:
            print("down")
            trial_count += 1
            trial_left -= 1
            
        elif int(question) < answer:
            print("up")
            trial_count += 1
            trial_left -= 1

        if trial_left == 0:
            print(f"아쉽습니다. 정답은 {answer}였습니다.")
            break        
    except ValueError:
        question = print("숫자를 입력해주세요")