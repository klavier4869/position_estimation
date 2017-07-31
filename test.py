import numpy as np
import matplotlib.pyplot as plt
from pdsolv import load_pdsolv

train, valid, test = load_pdsolv(normalize=False, flatten=False)

plt.imshow(test[0][1])
print(test[1][1])
plt.show()
