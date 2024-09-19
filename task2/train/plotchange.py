import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


input_path = r'C:\Users\HMRda\Desktop\pytorch\OCT\task2\train\origional'
output_path = r'C:\Users\HMRda\Desktop\pytorch\OCT\task2\train\output'

num = 0

for img_name in os.listdir(input_path):

    img_path = os.path.join(input_path,img_name)

    img = cv2.imread(img_path)

    # 获取图像中心
    center = (img.shape[1] // 2, img.shape[0] // 2)

    # 计算最大半径
    max_radius = min(center[0], center[1])

    # 执行极坐标到直角坐标的转换
    rect_img = cv2.warpPolar(img, (img.shape[1], img.shape[0]), center, max_radius, cv2.WARP_POLAR_LINEAR)

    num = num + 1

    # 保存结果
    cv2.imwrite(os.path.join(output_path, '{:06d}.jpg'.format(num)), rect_img)


