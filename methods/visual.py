import numpy as np

# def check(a):
#     return [[element if '0' not in element else '0' for element in row] for row in a]


def matmul(a, b):
    result = []
    for i in range(len(a)):
        row = []
        for j in range(len(b[0])):
            element = '+'.join([f'{a[i][k]}{b[k][j]}' if a[i][k]!='0' and b[k][j]!='0' else '0' for k in range(len(b))])
            row.append(element)
        result.append(row)
    return result

def element_wise(a, b):
    return [[f'{a[i][j]}{b[i][j]}' if a[i][j]!='0' and b[i][j]!='0' else '0'  for j in range(len(a[0]))] for i in range(len(a))]

def transpose(a):
    return [[a[j][i] for j in range(len(a))] for i in range(len(a[0]))]

def visualize_matrix(matrix):
    max_len = max(len(str(element)) for row in matrix for element in row)
    for row in matrix:
        print(' '.join(str(element).center(max_len) for element in row))
    print()

# 示例使用
# A = [['a11', 'a12'], ['a21', 'a22']]
# B = [['b11', 'b12'], ['b21', 'b22']]

# print("矩阵 A:")
# visualize_matrix(A)

# print("矩阵 B:")
# visualize_matrix(B)

# print("矩阵乘法 A * B:")
# result_matmul = matmul(A, B)
# visualize_matrix(result_matmul)

# print("逐元素乘法 A .* B:")
# result_element_wise = element_wise(A, B)
# visualize_matrix(result_element_wise)

# print("矩阵 A 的转置:")
# result_transpose = transpose(A)
# visualize_matrix(result_transpose)

def generate(name, size):
    rows, cols = size
    return [[f"{name}{i+1}{j+1}" for j in range(cols)] for i in range(rows)]

B = generate('b', [4,2])
C = generate('c', [4,2])
V = generate('v', [4,2])

visualize_matrix(B)
visualize_matrix(C)
visualize_matrix(V)

mask = [['1','0','0','0'], ['w','1','0','0'], ['w^2', 'w', '1', '0'], ['w^3', 'w^2', 'w', '1']]
ans = matmul(element_wise(matmul(B, transpose(C)), mask), V)
visualize_matrix(ans)