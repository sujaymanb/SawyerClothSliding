import numpy as np

lower = np.array([0,4,0])
upper = np.array([5,5,5])

test = np.array([3,4,3])

result_low = (test >= lower)
print("lower test: ", result_low)
result_upp = (test <= upper)
print("upper test: ", result_upp)
result = result_low * result_upp
print("bool sum", result)
print("result sum", result.prod())