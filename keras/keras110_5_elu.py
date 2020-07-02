import numpy as np
import matplotlib.pyplot as plt

a = 0.2
x = np.arange(-5,5,0.1)
y = [x if x>0 else a*(np.exp(x)-1) for x in x]

plt.plot(x, y)
plt.grid()
plt.show()