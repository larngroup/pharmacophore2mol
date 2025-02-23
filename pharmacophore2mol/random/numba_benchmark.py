import numpy as np
import time
from numba import njit

# Without JIT (Python version)
def compute_no_jit(values, threshold):
    valid_values = values[values != -1]
    if valid_values.size > 0:
        min_value = np.min(valid_values)
        return 1 if min_value >= threshold else 0
    else:
        return 0

# With JIT (Numba version)
@njit(parallel=True)
def compute_jit(values, threshold):
    valid_values = values[values != -1]
    if valid_values.size > 0:
        min_value = np.min(valid_values)
        return 1 if min_value >= threshold else 0
    else:
        return 0

# Example input
lists_per_pixel = np.random.randint(-1, 10, size=(3, 100, 100, 100, 8), dtype=np.int32)
threshold = 5

# Timing without JIT
start_time = time.time()
for _ in range(100):  # Run the function multiple times to get accurate timing
    np.apply_along_axis(compute_no_jit, -1, lists_per_pixel, threshold=threshold)
end_time = time.time()
print(f"Without JIT: {end_time - start_time} seconds")

# Timing with JIT
start_time = time.time()
for _ in range(100):  # Run the function multiple times to get accurate timing
    np.apply_along_axis(compute_jit, -1, lists_per_pixel, threshold=threshold)
end_time = time.time()
print(f"With JIT: {end_time - start_time} seconds")
