# coding:utf-8

import cv2
import numpy as np
import pylab as pl
from PIL import Image
import io

color = [[0, 255, 0], [50, 255, 0], [100, 255, 0], [150, 255, 0], [200, 255, 0], [255, 255, 0], [255, 200, 0],
         [255, 150, 0], [255, 100, 0], [255, 50, 0], [255, 25, 0],

         [255, 0, 0], [255, 0, 50], [255, 0, 100], [255, 0, 150], [255, 0, 200], [255, 0, 255], [200, 0, 255],
         [150, 0, 255], [100, 0, 255], [50, 0, 255], [5, 0, 255],

         [0, 0, 255], [0, 50, 255], [0, 100, 255], [0, 150, 255], [0, 200, 255], [0, 255, 255],
         [0, 255, 200], [0, 255, 150], [0, 255, 100], [0, 255, 50]]

IMAGE_DIR = 'F:/Tensorflow_models/models/research/deeplab/hair-output/segmentation_results/000000_image.png'
IMAGE_MASK_DIR = 'F:/Tensorflow_models/models/research/deeplab/hair-output/segmentation_results/000000_prediction.png'
# IMAGE_DIR = 'test.png'
# IMAGE_MASK_DIR = 'test_mask.png'
W = 100


# 构建Gabor滤波器
# returns a list of kernels in several orientations
def build_filters():

    filters = []

    for theta in np.arange(0, np.pi, np.pi / 180):  # gabor方向 0到180度之间均匀选取180个卷积核

            kern = cv2.getGaborKernel((0, 0), 1.8, theta, 4, 0.75, 0, ktype=cv2.CV_32F)

            # kern /= 1.5 * kern.sum()   # ?????

            filters.append(kern)

    return filters


def get_gabor(image, filters):

    image_gray = image.convert('L')  # 输入图像转化成灰度图
    img_ndarray = np.asarray(image_gray)
    res = []  # 滤波结果 180个图
    for i in range(len(filters)):
        res1 = cv2.filter2D(img_ndarray, cv2.CV_8UC3, filters[i])
        res.append(np.asarray(res1))
    return res


def get_theta(image_gabor, image_mask):

    c = image_mask.size[0]  # 图像的列数
    r = image_mask.size[1]  # 图像的行数
    num = 180  # 卷积核的个数

    image_mask_array = np.array(image_mask)
    res = np.array(image_mask.convert('L'))

    for i in range(r):
        for j in range(c):
            max = image_gabor[0][i, j]
            t = 0
            for k in range(num):
                if max < image_gabor[k][i, j]:
                    max = image_gabor[k][i, j]
                    t = k
            if image_mask_array[i, j][0] == 128:
                res[i, j] = t
            else:
                res[i, j] = 255
    return res


def get_w(image_gabor, image_theta):

    r = image_theta.shape[0]  # 图像的列数
    c = image_theta.shape[1]  # 图像的行数
    num = 180  # 卷积核的个数

    image_array = np.array(image_theta)
    res = np.array(image_theta)

    tot_m = 0.0
    for i in range(r):
        for j in range(c):
            if image_array[i, j] == 255:
                res[i, j] = 0
            else:
                tot = 0
                theta = 1.0 * image_array[i, j] * np.pi / 180.0
                for k in range(num):
                    theta1 = 1.0 * k * np.pi / 180.0
                    d = abs(theta - theta1)
                    d1 = abs(d - np.pi)
                    d2 = abs(d + np.pi)

                    d = min(d, d1)
                    d = min(d, d2)

                    f_theta1 = image_gabor[k][i, j]
                    f_theta = image_gabor[image_array[i, j]][i, j]

                    tmp = pow((f_theta - f_theta1), 2)

                    tmp = pow(d * tmp, 0.5)

                    tot += tmp
                tot_m = max(tot_m, tot)
                res[i, j] = tot / 180
    print(tot_m)
    return res


def get_theta_1(image_theta, image_w):

    r = image_theta.shape[0]  # 图像的列数
    c = image_theta.shape[1]  # 图像的行数

    image_array = np.array(image_theta)
    res = np.array(image_theta)

    for i in range(r):
        for j in range(c):
            if image_array[i, j] == 255:
                res[i, j] = 0
            else:
                if image_w[i, j] > W:
                    res[i, j] = image_array[i, j]
                else:
                    res[i, j] = 0

    return res


def get_real(image, image_theta, image_mask):

    r = image_theta.shape[0]  # 图像的列数
    c = image_theta.shape[1]  # 图像的行数
    image_theta_array = np.array(image_theta)
    image_mask_array = np.array(image_mask)
    res = np.array(image)

    for i in range(r):
        for j in range(c):
            if image_theta_array[i, j] != 0:
                theta = 1.0 * image_theta_array[i, j] / 180.0
                # print(theta)
                a = np.array([1, 0, 0])
                b = np.array([0, 0, 1])
                res[i, j] = (theta * a + (1 - theta) * b) * 255
            else:
                res[i, j] = [0, 0, 0]
    return res


if __name__ == '__main__':

    img = Image.open(IMAGE_DIR)
    img = img.resize((256, 256), Image.ANTIALIAS)
    pl.imshow(img)
    pl.show()
    img_mask = Image.open(IMAGE_MASK_DIR)
    img_mask = img_mask.resize((256, 256), Image.ANTIALIAS)

    filters = build_filters()

    # 1
    img_gabor = get_gabor(img, filters)

    img_theta = get_theta(img_gabor, img_mask)

    img_w = get_w(img_gabor, img_theta)  # 第一次得到的w，作为下一个gabor的输入

    img_theta_1 = get_theta_1(img_theta, img_w)  # 输出的色彩图

    img_real = get_real(img, img_theta_1, img_mask)

    img_real1 = Image.fromarray(img_real.astype('uint8')).convert('RGB')

    pl.imshow(img_real1)
    pl.show()

    # 2
    img_w1 = Image.fromarray(img_w.astype('uint8')).convert('RGB')

    img_gabor = get_gabor(img_w1, filters)

    img_theta = get_theta(img_gabor, img_mask)

    img_w = get_w(img_gabor, img_theta)

    img_theta_1 = get_theta_1(img_theta, img_w)

    # 3

    img_w1 = Image.fromarray(img_w.astype('uint8')).convert('RGB')

    img_gabor = get_gabor(img_w1, filters)

    img_theta = get_theta(img_gabor, img_mask)

    img_w = get_w(img_gabor, img_theta)

    img_theta_1 = get_theta_1(img_theta, img_w)




    img_real = get_real(img, img_theta_1, img_mask)

    img_real1 = Image.fromarray(img_real.astype('uint8')).convert('RGB')

    # pl.imshow(img_real1)
    # pl.show()

    img_real1.save("F:/HairNet_save/HairNet/HairNet/network/data/real/test.png")

    # pl.imshow(img_real1)
    # pl.show()




