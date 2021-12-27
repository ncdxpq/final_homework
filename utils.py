import numpy as np
import cv2
import copy
from min_ercheng import get_nihe_pingmian
import math


# img = cv2.imread('E:/Homework_final/pictures/apple_gray.jpg',0)  # 0代表获得灰度图，因为解释器不知道图片是不是灰度图


# 欧式距离
def eucliDist(A, B):
    """计算两点之间的欧式距离
    eg:x:X = np.array([1,2,3,4])，Y = np.array([0,1,2,3])"""
    return np.sqrt(sum(np.power((A - B), 2)))


def get_gray_photo(load_path, save_path, newH, newW):
    """ 读取图像并且将其转化为灰度图，返回灰度图的像素值。
    同时将图片存储在filepath"""
    # 读取图像
    origin = cv2.imread(load_path)
    # 图像缩放
    origin = cv2.resize(origin, (newW, newH))
    # 转成灰度图
    img_gray = cv2.cvtColor(origin, cv2.COLOR_RGB2GRAY)
    # cv2.imshow("title", img_gray)  # 展示灰度图
    cv2.imwrite(save_path, img_gray)  # 保存
    # print('转换之后的图的维度:', img_gray.shape)  # 查看灰度图的维度
    return img_gray


def show_pixel_photo(pixel):
    '''
    传入已经经过处理的灰度图像的像素点（500*500），画出图
    '''
    cv2.imshow("title", pixel)  # 展示灰度图
    print('维度:', pixel.shape)  # 查看灰度图的维度
    cv2.waitKey(0)


def get_pixel(img, name):
    """
    获得图片的像素值（100*100）,并存储到txt文档中
    """
    ARRS = []
    f = open(name, 'w+')
    for i in range(100):
        jointsFrame = img[i]  # 每行
        ARRS.append(jointsFrame)
        for Ji in range(100):
            strNum = str(jointsFrame[Ji])
            f.write(strNum)
            f.write(' ')
        f.write('\n')
    f.close()


def show_seed_on_oriPhoto(flag_of_seed_matrix, gray_photoPath, savepath):
    """
    原图已经被转化为灰度图并且缩小，这里载入本地的灰色图片，
    直接修改种子点的像素值，使其拥有颜色，便可查看到种子点的位置。
    然后将有种子点的地方设置成有颜色即可
    """
    # 读取图像
    # print('-----------------------')
    # print('开始展示种子点在灰度图的位置：')
    img = cv2.imread(gray_photoPath)
    # 直接在原图像上修改像素点，查看种子点的位置
    for i in range(len(flag_of_seed_matrix[0])):
        for j in range(len(flag_of_seed_matrix[1])):
            if flag_of_seed_matrix[i][j] == 1:  # 是种子点
                img[i][j][:] = [0, 255, 0]  # 设置种子点为绿色

    # cv2.imshow("seed_position", img)  # 展示灰度图
    cv2.imwrite(savepath, img)  # 保存
    # print('种子点位置图片已经保存在{}'.format(savepath))
    # cv2.waitKey(0)


def get_euclidean_distance_and_class(flag_of_seed_matrix):
    '''传入当前的种子点矩阵，仅根据位置返回所有点的最短的欧氏距离以及种子点序号'''
    init_seed_class_matrix = copy.deepcopy(flag_of_seed_matrix)
    seed_index = []
    incerment = 0
    for i in range(len(flag_of_seed_matrix[0])):
        for j in range(len(flag_of_seed_matrix[1])):
            if init_seed_class_matrix[i][j] != 0:  # 是个种子点
                init_seed_class_matrix[i][j] += incerment
                incerment += 1
                seed_index.append([i, j])

    get_pixel(init_seed_class_matrix, 'seed.txt')
    # print('---------------种子点类别和位置展示：---------------')
    # print(init_seed_class_matrix)  # 种子点标记矩阵数字由1变成对应的类别数

    # 初始化最小距离矩阵
    min_euclidean_distance = np.zeros((len(flag_of_seed_matrix[0]), len(flag_of_seed_matrix[1])))
    # 初始化对应的最小距离的类别
    min_euclidean_distance_class = np.zeros((len(flag_of_seed_matrix[0]), len(flag_of_seed_matrix[1])))

    # 迭代求最小的欧氏距离 # todo 优化：从种子点为开始，以半径的方式搜索最近的种子点位置
    for w in range(len(flag_of_seed_matrix[0])):
        for h in range(len(flag_of_seed_matrix[1])):
            if flag_of_seed_matrix[w][h] != 0:  # 该点是种子点
                min_euclidean_distance[w][h] = 0  # 对种子点来说，最短的欧氏距离为0
                min_euclidean_distance_class[w][h] = init_seed_class_matrix[w][h]  # 种子点类别
            else:
                # 该点非种子点，循环求出其与其他种子点最短的欧氏距离，并且获得该种子点的类别（可优化）
                for a in range(len(flag_of_seed_matrix[0])):
                    for b in range(len(flag_of_seed_matrix[1])):
                        if flag_of_seed_matrix[a][b] != 0:  # 种子点
                            if min_euclidean_distance[w][h] == 0:  # 还没赋值，直接给类别并且计算
                                min_euclidean_distance[w][h] = eucliDist(np.array([w, h]),
                                                                         np.array([a, b]))
                                min_euclidean_distance_class[w][h] = init_seed_class_matrix[a][b]
                            else:  # 已经有值了，比谁最小
                                if eucliDist(np.array([w, h]), np.array([a, b])) < min_euclidean_distance[w][h]:
                                    min_euclidean_distance[w][h] = eucliDist(np.array([w, h]),
                                                                             np.array([a, b]))
                                    min_euclidean_distance_class[w][h] = init_seed_class_matrix[a][b]

    return min_euclidean_distance, min_euclidean_distance_class, seed_index


def find_border(class_Matrix):
    """
    发现边界点，返回边界点的坐标[x][y]方便在原图上画出来
    :param class_Matrix:
    :return: 二维数组，代表种子点的坐标eg:[[1, 17], [1, 17], [1, 20], [1, 20], [1, 43],......
    """
    point = []
    classmatrix = copy.deepcopy(class_Matrix)
    for i in range(1, len(class_Matrix[0]) - 1):  # 图像边缘的点自动忽略，不算边界点
        for j in range(1, len(class_Matrix[1]) - 1):  # 图像边缘的点自动忽略，不算边界点
            if classmatrix[i][j] != classmatrix[i][j + 1]:  # 两个点都是边界点
                temp = []
                temp.append(i)
                temp.append(j)
                temp1 = []
                temp1.append(i)
                temp1.append(j)
                point.append(temp)
                point.append(temp1)
    return point


def show_seed_and_border_on_ori_img(flag_of_seed_matrix, gray_photoPath, savepath, classMatrix):
    """
    原图已经被转化为灰度图并且缩小，这里载入本地的灰色图片，
    直接修改种子点的像素值，使其拥有颜色，便可查看到种子点的位置。
    然后将有种子点的地方设置成有颜色即可。同时将边界设置为红色
    """
    # 读取图像
    # print('-----------------------')
    # print('开始展示种子点在灰度图的位置：')
    img = cv2.imread(gray_photoPath)
    # 直接在原图像上修改像素点，查看种子点的位置
    for i in range(len(flag_of_seed_matrix[0])):
        for j in range(len(flag_of_seed_matrix[1])):
            if flag_of_seed_matrix[i][j] == 1:  # 是种子点
                img[i][j][:] = [0, 255, 0]  # 设置种子点为绿色
    # 遍历所有边界点，在原图上展示出来
    for i in range(len(classMatrix)):
        x, y = classMatrix[i][0], classMatrix[i][1]
        img[x][y][:] = [255, 255, 0]
    # cv2.imshow("seed_position", img)  # 展示灰度图
    cv2.imwrite(savepath, img)  # 保存
    # print('种子点和边框位置图片已经保存在{}'.format(savepath))
    # cv2.waitKey(0)


# 调用最小二乘法，获得逼近的超平面的系数，展示每个区域内平面的逼近结果
def show_ordinary_least_squares_result(min_euclidean_distance_class, origin, classnumber, savepath):
    """
    x_y_z：是x,y,z的集合，eg:[[[0, 48, 255], [0, 49, 255], [0, 50, 255], [0, 51, 255], [0, 52, 255], [0, 53, 255],
    返回拟合的系数矩阵，和拟合的像素点，方便计算误差
    """
    x_y_z = [[] for z in range(classnumber)]
    for i in range(len(min_euclidean_distance_class[0])):
        for j in range(len(min_euclidean_distance_class[1])):
            # min_euc从1开始的，要-1才行
            # 对应的类别，加入x,y,z坐标
            pixel = origin[i][j]
            x_y_z[int(min_euclidean_distance_class[i][j]) - 1].append([i, j, pixel])

    # len(x_y_z)=100,即100个类对应100个种子点的集合
    xishu = []  # 第0个元素对应class=1的种子点区域超平面的系数
    for i in range(len(x_y_z)):
        # 获得拟合的系数z=ax+by+c中的a,b,c
        a, b, c = get_nihe_pingmian(x_y_z[i])
        xishu.append([a, b, c])
    pixel = copy.deepcopy(origin)  # 得到新的像素点，画出来
    for i in range(len(min_euclidean_distance_class[0])):
        for j in range(len(min_euclidean_distance_class[1])):
            # min_euclidean_distance_class是从1开始的，需要-1
            tempxishu = xishu[int(min_euclidean_distance_class[i][j] - 1)]
            z = int(tempxishu[0] * i + tempxishu[1] * j + tempxishu[2])  # 获得超平面拟合的像素点z=ax+by+c
            pixel[i][j] = z
    # print('最小二乘拟合的区域内平面结果已经保存在{}'.format(savepath))
    cv2.imwrite(savepath, pixel)
    return xishu, pixel


def compute_loss(origin, newpixel):
    # 计算最小二乘法拟合的超平面与原图像直接的loss
    loss = 0
    for i in range(len(newpixel[0])):
        for j in range(len(newpixel[1])):
            loss += (int(origin[i][j]) - int(newpixel[i][j])) ** 2
    return loss


def get_common_border(class_Matrix, classnumber):
    # get_pixel(class_Matrix,'class.txt')
    """传入类别矩阵，返回当前类别种子点的边界以及相邻种子点区域的边界，
    比如border_x_y_a_b[0]代表第一类种子点的边界及其邻接边界，如[1,3,2,3]即
    意为[1,3]与[2,3]是临界点"""
    border_x_y_a_b = [[] for i in range(classnumber)]  # class类别要-1,因为此数组从0开始

    for i in range(classnumber):
        for j in range(classnumber):
            # 找右下角的邻居
            if (i + 1 < classnumber) and (j + 1) < classnumber and (class_Matrix[i + 1][j + 1] != class_Matrix[i][j]):
                border_x_y_a_b[int(class_Matrix[i][j]) - 1].append([i, j, i + 1, j + 1])
                # neighbor_x_y_class[int(class_Matrix[i][j]) - 1].append([i + 1, j + 1, int(class_Matrix[i + 1][j + 1])])
            # 右上角
            if (i + 1 < classnumber) and (j - 1) > 0 and (class_Matrix[i + 1][j - 1] != class_Matrix[i][j]):
                border_x_y_a_b[int(class_Matrix[i][j]) - 1].append([i, j, i + 1, j - 1])
                # neighbor_x_y_class[int(class_Matrix[i][j]) - 1].append([i + 1, j - 1, int(class_Matrix[i + 1][j - 1])])
            # 左上角
            if (i - 1 > 0) and (j - 1 > 0) and (class_Matrix[i - 1][j - 1] != class_Matrix[i][j]):
                border_x_y_a_b[int(class_Matrix[i][j]) - 1].append([i, j, i - 1, j - 1])
                # neighbor_x_y_class[int(class_Matrix[i][j]) - 1].append([i - 1, j - 1, int(class_Matrix[i - 1][j - 1])])
            # 左下角
            if (i - 1 > 0) and (j + 1) < (classnumber) and (class_Matrix[i - 1][j + 1] != class_Matrix[i][j]):
                border_x_y_a_b[int(class_Matrix[i][j]) - 1].append([i, j, i - 1, j + 1])
                # neighbor_x_y_class[int(class_Matrix[i][j]) - 1].append([i - 1, j + 1, int(class_Matrix[i - 1][j + 1])])
            # 上
            if (j - 1) > 0 and (class_Matrix[i][j - 1] != class_Matrix[i][j]):
                border_x_y_a_b[int(class_Matrix[i][j]) - 1].append([i, j, i, j - 1])
                # neighbor_x_y_class[int(class_Matrix[i][j]) - 1].append([i, j - 1, int(class_Matrix[i][j - 1])])
            # 下
            if (j + 1) < classnumber and (class_Matrix[i][j + 1] != class_Matrix[i][j]):
                border_x_y_a_b[int(class_Matrix[i][j]) - 1].append([i, j, i, j + 1])
                # neighbor_x_y_class[int(class_Matrix[i][j]) - 1].append([i, j + 1, int(class_Matrix[i][j + 1])])
            # 左
            if (i - 1) > 0 and (class_Matrix[i - 1][j] != class_Matrix[i][j]):
                border_x_y_a_b[int(class_Matrix[i][j]) - 1].append([i, j, i - 1, j])
                # neighbor_x_y_class[int(class_Matrix[i][j]) - 1].append([i - 1, j, int(class_Matrix[i - 1][j])])
            # 右
            if (i + 1) < (classnumber) and (class_Matrix[i + 1][j] != class_Matrix[i][j]):
                border_x_y_a_b[int(class_Matrix[i][j]) - 1].append([i, j, i + 1, j])
                # neighbor_x_y_class[int(class_Matrix[i][j]) - 1].append([i + 1, j, int(class_Matrix[i + 1][j])])
    # for i in range(len(border_x_y)):
    #     # 去除重复的元素
    #     border_x_y[i] = list(set(border_x_y[i]))
    return border_x_y_a_b


# 坐标相减
def x_pi(x, pi):
    c = [x[i] - pi[i] for i in range(0, len(x))]
    return c


# 向量模长
def mo_pi_pj(pi, pj):
    a = x_pi(pi, pj)
    mo = math.sqrt(a[0] ** 2 + a[1] ** 2)
    return mo


# 向量与数除法
def div(vector, mo):
    c = [(vector[i] / mo) for i in range(0, len(vector))]
    return c


# 向量与数乘法
def mul(vector, num):
    c = [(vector[i] * num) for i in range(0, len(vector))]
    return c


# 获得每个种子点的梯度，二维数组
def get_gradient(origin, newpixel, commonborder_x_y_a_b, seed_index, num_of_seed, classMatrix):
    seed_gradient = []
    for index, items in enumerate(commonborder_x_y_a_b):  # 每类种子点的区域都求一次梯度
        gradient_pair = [0, 0]
        for k in range(len(items)):  # 每个xyab
            x, y, a, b = items[k][0], items[k][1], items[k][2], items[k][3]
            # 代入目标函数对种子点的梯度公式
            first = (int(origin[x][y]) - int(newpixel[x][y])) ** 2
            second = (int(origin[x][y]) - int(newpixel[a][b])) ** 2
            pi = seed_index[int(classMatrix[x][y]) - 1]  # 获得了当前区域的种子点的坐标
            pj = seed_index[int(classMatrix[a][b]) - 1]  # 获得了与当前区域邻接的种子点的坐标
            x_pi1 = x_pi([x, y], pi)
            mo_pi_pj1 = mo_pi_pj(pi, pj)
            third = div(x_pi1, mo_pi_pj1)
            result = mul(third, first - second)
            gradient_pair[0] += result[0]
            gradient_pair[1] += result[1]
        seed_gradient.append(gradient_pair)
    return seed_gradient


def gradient_descent(gradient_every_seed, seed_index, lr):
    new_seed_position = []
    for r in range(len(seed_index)):  # 对每个种子点都进行梯度下降
        x_grad = gradient_every_seed[r][0]  # 该种子点对x的梯度
        y_grad = gradient_every_seed[r][1]  # 该种子点对y的梯度
        x = seed_index[r][0]  # x坐标
        y = seed_index[r][1]  # y坐标
        x -= x_grad * lr
        y -= y_grad * lr
        # 先四舍五入，再将超范围的x和y放到边界处
        x = round(x)
        y = round(y)
        if x < 0:
            x = 0
        if x > len(seed_index) - 1:
            x = len(seed_index) - 1
        if y < 0:
            y = 0
        if y > len(seed_index) - 1:
            y = len(seed_index) - 1

        new_seed_position.append([x, y])
    return new_seed_position


def random_gradient_descent(gradient_every_seed, seed_index, lr, random_rate):
    rand_num = int(random_rate * len(gradient_every_seed))
    temp_index = np.array([1] * rand_num + [0] * (len(gradient_every_seed) - rand_num))
    np.random.shuffle(temp_index)  # 随机打乱顺序

    new_seed_position = []
    for r in range(len(seed_index)):  # 对每个种子点都进行梯度下降
        if int(temp_index[r]) == 1:  # 该种子点不进行梯度下降
            x = seed_index[r][0]
            y = seed_index[r][1]
            new_seed_position.append([x, y])
            continue

        x_grad = gradient_every_seed[r][0]  # 该种子点对x的梯度
        y_grad = gradient_every_seed[r][1]  # 该种子点对y的梯度
        x = seed_index[r][0]  # x坐标
        y = seed_index[r][1]  # y坐标
        x -= x_grad * lr
        y -= y_grad * lr
        # 先四舍五入，再将超范围的x和y放到边界处
        x = round(x)
        y = round(y)
        if x < 0:
            x = 0
        if x > len(seed_index) - 1:
            x = len(seed_index) - 1
        if y < 0:
            y = 0
        if y > len(seed_index) - 1:
            y = len(seed_index) - 1
        new_seed_position.append([x, y])
    return new_seed_position


# 得到初始的种子点标记矩阵
def get_new_flag_of_seed_matrix(new_position, h=100, w=100):
    a = np.array([0] * (h * w))
    temp = a.reshape((w, h))
    for pair_position in new_position:
        x = pair_position[0]
        y = pair_position[1]
        temp[x][y] = 1
    # get_pixel(temp,'temp.txt')
    return temp
