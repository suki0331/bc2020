year = int(input())

def is_leap_year(year):
    result = int()
    if year % 4 == 0:
        if year % 100 != 0 or year % 400 == 0:
            result = 1
        else:
            result = 0
    print(result)

if __name__ == "__main__":
    is_leap_year(year)