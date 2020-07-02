import numpy as np
import matplotlib.pyplot as plt
a = 0.2
ALPHA = 1.6732632423543772848170429916717
LAMBDA = 1.0507009873554804934193349852946
x = np.arange(-5, 5, 0.1)
y = [LAMBDA*(x if x>0 else ALPHA*np.exp(x)-ALPHA) for x in x]

plt.plot(x,y)
plt.grid()
plt.show()