import cv2
import numpy as np
from utils import get_gray_photo
from utils import get_pixel  # 自己定义的查看像素点的函数
from utils import show_seed_on_oriPhoto
import copy
from utils import get_euclidean_distance_and_class
from utils import show_pixel_photo
from utils import find_border
from utils import show_seed_and_border_on_ori_img
from utils import show_ordinary_least_squares_result
from utils import compute_loss
from utils import get_common_border
from utils import get_gradient
from utils import random_gradient_descent
from utils import gradient_descent
from utils import get_new_flag_of_seed_matrix

# ======准备工作======

# 设置缩放系数图片，新的图片大小
newh = 100
neww = 100

# 将图像转换为更小像素的灰度图像并且保存，get_gray_photo返回该图片灰度值
origin = get_gray_photo('E:/Homework_final/pictures/apple.jpg',
                        'E:/Homework_final/pictures/gray_apple.jpg',
                        newH=newh,
                        newW=neww)

# 输出原图像大小信息
h, w = origin.shape
# print("原始图像大小为 ：{}*{},此时已经转化为灰度图像，查看picture文件获得该图像！！！".format(h, w))

# 随机初始化种子点的个数
num_of_seed = int(input("\n请输入种子点个数 ： "))  # 100个比较好
epochs = int(input("\n请输入迭代轮数 ： "))  # 15轮左右

# 初始化种子点标记矩阵
temp_matrix = np.array([1] * num_of_seed + [0] * (h * w - num_of_seed))  # 辅助数组，用于生成种子点标记矩阵
np.random.shuffle(temp_matrix)  # 随机打乱顺序
flag_of_seed_matrix = temp_matrix.reshape((w, h))  # 种子点标记矩阵，1代表该像素点为种子点
print('******开始利用随机梯度下降法优化种子点位置******')

# 开始利用梯度下降法优化种子点位置
for epoch in range(epochs):
    # 查看初始化的种子点矩阵，保存在.txt文件中
    # get_pixel(flag_of_seed_matrix, name='init_flag_of_seed_matrix.txt')

    # 在缩放的灰度图上查看一下初始的种子点的位置,并且保存在savepath
    show_seed_on_oriPhoto(flag_of_seed_matrix,
                          'E:/Homework_final/pictures/gray_apple.jpg',
                          savepath='E:/Homework_final/pictures/init_seed_apple.jpg')

    # 计算各个像素点与种子点直接的最短欧式距离并且分类，每个种子点是一类，便获得了题干初始状态。同时获得种子点位置:1类在第0个位置
    # print('开始获得题干的网格图...')
    min_euclidean_distance, min_euclidean_distance_class, seed_index = get_euclidean_distance_and_class(
        flag_of_seed_matrix)

    # print('-----------------每个点代表该像素点距离哪个种子点类最近--------------')
    # print(min_euclidean_distance_class)

    # 获得初始的分类图，不超过254个种子点，这样可以直接画出灰度图，比较简单
    # print('初始化题干的网格图，存放在{}成功！'.format('E:/Homework_final/pictures/init_min_distance_class{}.jpg'.format(epoch)))
    cv2.imwrite('E:/Homework_final/pictures/init_min_distance_class{}.jpg'.format(epoch), min_euclidean_distance_class)

    # 获得边界点，格式为：eg:[[1, 17], [1, 17], [1, 20], [1, 20], [1, 43],......
    border = find_border(min_euclidean_distance_class)

    # 同时在原图上显示边界点和边框
    show_seed_and_border_on_ori_img(flag_of_seed_matrix,
                                    'E:/Homework_final/pictures/gray_apple.jpg',
                                    savepath='E:/Homework_final/pictures/init_seed_and_border_apple{}.jpg'.format(
                                        epoch),
                                    classMatrix=border)

    # ========================开始最优化部分=============================
    # 最小二乘法逼近每个区域内的超平面，并且存储逼近的结果。
    erchengxishu, newpixel = show_ordinary_least_squares_result(min_euclidean_distance_class,
                                                                origin,
                                                                classnumber=num_of_seed,
                                                                savepath='E:/Homework_final/pictures/erchengnihe{}.jpg'.format(
                                                                    epoch))

    # 计算原图与最小二乘法拟合的loss
    loss = compute_loss(origin, newpixel)  # 30974613

    print('第{}轮迭代的loss是{}'.format(epoch + 1, loss))

    # 得到拥有共同边界的种子点序号集合，[[],[]...二维列表，第一个为第一个类的边界
    commonborder_x_y_a_b = get_common_border(min_euclidean_distance_class, classnumber=num_of_seed)

    # 获得每个种子点的梯度，二维数组，
    gradient_every_seed = get_gradient(origin, newpixel,
                                       commonborder_x_y_a_b,
                                       seed_index,
                                       num_of_seed=num_of_seed,
                                       classMatrix=min_euclidean_distance_class)
    # print(gradient_every_seed)  [[-3.605211308741203, -4.070093083854642], [16.44279709470809, 14.634815521090971],

    # ==================梯度下降,求出新的种子点位置=======================
    # new_seed_position = gradient_descent(gradient_every_seed, seed_index, lr=0.000001)
    # print(new_seed_position)  # 格式：[[0, 11], [0, 18], [1, 79], ...

    # ==================随机梯度下降,求出新的种子点位置=======================
    # random_rate：随机梯度下降的种子点比例
    new_seed_position = random_gradient_descent(gradient_every_seed, seed_index, lr=0.000001,random_rate=0.5)

    # ==================本次迭代结束，画出新的初始点的位置====================
    flag_of_seed_matrix = get_new_flag_of_seed_matrix(new_seed_position)
