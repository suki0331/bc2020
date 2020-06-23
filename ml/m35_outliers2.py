# practice
# write a function that finds outliers by iterating columns
import numpy as np


def find_outlier(data):
    li = []
    for i in range(data.shape[0]):
        tem_data = data[i]
        quartile_1, quartile_3, = np.percentile(tem_data, [25, 75])
        print(f"q1: {quartile_1}")
        print(f"q3: {quartile_3}")
        iqr = quartile_3 - quartile_1
        lb = quartile_1 - (iqr * 1.5)
        ub = quartile_3 + (iqr * 1.5)
        d_out = np.where((tem_data>ub) | (tem_data<lb))
        # print(d_out)
        li.append(d_out)    
        # print(li)
    return li
a = np.array([[1,2,3,4,1000,6,7,5000,90,100],[2,3,4,2000,5,7,8,6000,80,200],[1,2,34,5,8,56,4]])

b = find_outlier(a)

print(b)
