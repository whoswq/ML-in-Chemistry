import numpy as np


x_array = np.array([[1, 2, 3], [4, 5, 6]])
y_array = x_array.reshape(-1)
print(y_array)
print(x_array)
y_array[0] = 1200
print(y_array)
print(x_array)