import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图片
img = cv2.imread('your_image.png')

# 获取图像中心
center = (img.shape[1] // 2, img.shape[0] // 2)

# 计算最大半径
max_radius = min(center[0], center[1])

# 执行极坐标到直角坐标的转换
rect_img = cv2.warpPolar(img, (img.shape[1], img.shape[0]), center, max_radius, cv2.WARP_POLAR_LINEAR)

# 保存结果
cv2.imwrite('rect_image.png', rect_img)

# 使用Matplotlib显示结果
plt.imshow(cv2.cvtColor(rect_img, cv2.COLOR_BGR2RGB))
plt.title('Rectangular Image')
plt.axis('off')
plt.show()
