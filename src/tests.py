import numpy as np
import antropy as ant

data = np.array([0.2, 0.7, 0.4, 0.9, 0.1, 0.5, 1, 8, 16, 12, 0.9, 0.7])

def lempelZivTest(data):
    # Para data = [0.2, 0.7, 0.4, 0.9, 0.1, 0.5, 1, 8, 16, 12, 0.9, 0.7]
    threshold = np.mean(data)
    print(threshold) # 3.4499999999999997
    binary_data = (data >= threshold) * 1
    print(binary_data) # [0 0 0 0 0 0 0 1 1 1 0 0]
    print(ant.lziv_complexity(binary_data, normalize=False)) # 4
    print(ant.lziv_complexity(binary_data, normalize=True))  # 1.195

lempelZivTest(data)