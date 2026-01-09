import numpy as np

# Creating a 2D NumPy array from a Python list
matrix = np.array([
    (10, 20, 30),
    (40, 50, 60)
])

print("Matrix:")
print(matrix)
print("Type:", type(matrix))

# Creating a 1D NumPy array
vector = np.array([5, 15, 25])

print("\nVector:")
print(vector)

# Checking data type of elements
print("\nVector data type:", vector.dtype)

# Checking dimensions
print("\nMatrix dimensions:", matrix.ndim)   # 2D array
print("Vector dimensions:", vector.ndim)     # 1D array

# Checking shape (rows, columns)
print("\nMatrix shape:", matrix.shape)
print("Vector shape:", vector.shape)

# Checking data type of matrix
print("\nMatrix data type:", matrix.dtype)

# Creating array with a specific data type (int16)
matrix_int16 = np.array([
    (7, 14, 21),
    (28, 35, 42)
], dtype='int16')

print("\nMatrix with int16 data type:")
print(matrix_int16)
print("Data type:", matrix_int16.dtype)

# Size of each element in bytes
print("\nItem size (bytes):", matrix_int16.itemsize)

# Changing data type to int32
matrix_int32 = np.array([
    (7, 14, 21),
    (28, 35, 42)
], dtype='int32')

print("\nItem size with int32:", matrix_int32.itemsize)

# Total number of elements
print("Total elements:", matrix_int32.size)

# Total memory usage
print("Total memory (manual):", matrix_int32.size * matrix_int32.itemsize)
print("Total memory (nbytes):", matrix_int32.nbytes)
