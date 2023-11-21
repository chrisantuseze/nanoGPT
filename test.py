import tensorflow as tf

# Assuming you have an existing matrix
existing_matrix = tf.constant([[1.9, 2, 3], [4, 5, 6], [7, 8, 9]])

# Get the shape of the existing matrix
matrix_shape = tf.shape(existing_matrix)

# Create a lower triangular matrix of ones based on the shape
lower_triangular_ones = tf.linalg.band_part(tf.ones(matrix_shape), -1, 0)

# Print the result
print("Existing Matrix:")
print(existing_matrix.numpy())
print("\nLower Triangular Matrix of Ones:")
print(lower_triangular_ones.numpy())
