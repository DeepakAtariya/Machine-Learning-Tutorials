from numpy import linalg as LA
import numpy as np
matrix = [
    [8,5,1,5,6],
    [5,3,8,7,5],
    [1,8,10,7,8],
    [5,7,7,8,9],
    [6,5,8,9,7]
]

matrix_w = np.array(matrix)

# print()
matrix_w_square = pow(matrix_w,2)
# matrix_w_square = LA.matrix_power(matrix_w,2)

print(matrix_w_square)

# eigenvalue , eigenvector = LA.eig(matrix_w)
eigenvalue , eigenvector = LA.eig(matrix_w_square)

print(pow(max(eigenvalue),0.5))
print(pow(16,0.5))

# f = [1,2,3,4]
# print(f)
