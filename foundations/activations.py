import numpy as np
import math
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        for i in range(z.size):
            x=z[i]
            x=1/(1+math.exp(-x))
            z[i]=np.round(x,5)
        return z

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        for i in range(z.size):
            x=z[i]
            x=max(0,x)
            z[i]=np.round(x,5)
        return z
