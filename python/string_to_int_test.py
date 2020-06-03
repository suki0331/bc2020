import numpy as np
import pandas as pd

a = ['12,331']

a[0] = int(a[0].replace(',',''))
print(a)
print(type(a))
print(type(a[0]))