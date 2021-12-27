import numpy as np


def get_matrix(x_y_z):
    x = []
    y = []
    z = []
    for i in range(len(x_y_z)):
        x.append(x_y_z[i][0])
        y.append(x_y_z[i][1])
        z.append(x_y_z[i][2])
    # 创建系数矩阵A和b
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    A = np.ones((len(x_y_z), 3))  # len(x_y_z)*3矩阵，全是1
    b = np.zeros((len(x_y_z), 1))  # len(x_y_z)*1矩阵，全是1
    for i in range(0, len(x_y_z)):  # A的前两列全部赋值为点x和y，第三列为1。b由z的值确定
        A[i, 0] = x[i]
        A[i, 1] = y[i]
        b[i, 0] = z[i]
    return A, b, x, y, z


def matrix_compute(A, b):
    """X=(AT*A)-1*AT*b直接求解"""
    # A_T = A.T  # 获得矩阵A的转置(3*188)
    # AT_A = np.dot(A_T, A)  # 矩阵乘法(3*188)*(188*3)=(3*3)
    X = np.linalg.lstsq(A, b, rcond=None)[0]
    # AT_A_reverse = np.linalg.inv(AT_A)  # 矩阵求逆(3*3)
    # AT_A_reverse_A_T = np.dot(AT_A_reverse, A_T)  # (3*3)*(3*188)=(3*188)
    # X = np.dot(AT_A_reverse_A_T, b)  # (3*188)*(188*1)=(3*1)
    # print('最小二乘法拟合的最终平面为：z = %.2f * x + %.2f * y + %.2f' % (X[0, 0], X[1, 0], X[2, 0]))
    return X  # 获得系数a,b,c


def get_nihe_pingmian(x_y_z):
    # 获得题干的函数z=2x+8y+10，将其系数填入矩阵以及获得该188个点
    A, b, x, y, z = get_matrix(x_y_z)
    X = matrix_compute(A, b)  # 最小二乘法运算，获得未知系数a,b,c

    return X[0, 0], X[1, 0], X[2, 0]
