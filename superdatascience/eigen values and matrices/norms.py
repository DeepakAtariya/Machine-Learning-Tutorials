import numpy as np

yzero = np.array([0,1,0,0,0])
b = np.array([1,0,0,0,0])

matrix = [
    [8,5,1,5,6],
    [5,3,8,7,5],
    [1,8,10,7,8],
    [5,7,7,8,9],
    [6,5,8,9,7]
]

matrix_w = np.array(matrix)

y1 = matrix_w*yzero+b
# print(y1)
l2norm = np.linalg.norm(y1,2)
l2normviafro = np.linalg.norm(y1,'fro')
print("y1 norm is "+str(l2normviafro))

y2 = matrix_w*y1+b
l2norm = np.linalg.norm(y2,2)
l2normviafro = np.linalg.norm(y2,'fro')

y2norm = l2normviafro

yzeronorm = np.linalg.norm(yzero,2)
print(yzeronorm)

print("||Y2||2/||Y0||2 is -"+str(y2norm/yzeronorm))
