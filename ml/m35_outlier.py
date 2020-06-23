import numpy as np

def outliers(data_out):
    quartile_1, quartile_3, = np.percentile(data_out, [25, 75])
    print(f"q1: {quartile_1}")
    print(f"q3: {quartile_3}")
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return data_out[np.where((data_out>upper_bound) | (data_out<lower_bound))]


a = np.array([1,2,3,4,1000,6,7,5000,90,100])
# np.where((data_out>upper_bound) | (data_out<lower_bound)) == a[4,7]
b = outliers(a)

print(b)

a = np.array([[1,2,3,4,1000,6,7,5000,90,100],[2,3,4,5,2000,7,8,6000,80,200]])
print(np.mean(a[0]))
print(a[0][1])
print(a[1][4])

out = np.empty([a.shape[0],1])
print(out.shape)