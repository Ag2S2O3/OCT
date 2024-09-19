import os
import cv2
import torch
import segmentation_models_pytorch as smp
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# 文件夹路径
label_folder = r'C:\Users\HMRda\Desktop\pytorch\OCT\task1\parameter\try\labels'  # 原始标签的文件夹
pred_folder = r'C:\Users\HMRda\Desktop\pytorch\OCT\task1\parameter\try\outputs'    # 模型预测结果的文件夹

# 获取文件名列表
label_files = sorted(os.listdir(label_folder))
pred_files = sorted(os.listdir(pred_folder))

# 初始化 TensorBoard
writer = SummaryWriter(log_dir=r'C:\Users\HMRda\Desktop\pytorch\OCT\task1\parameter\try')

# 初始化总和变量，用于计算平均值
iou_sum, acc_sum, dice_sum = 0, 0, 0
num_samples = len(label_files)

# 遍历每一对标签和预测图像
for i, (label_file, pred_file) in enumerate(zip(label_files, pred_files)):
    # 加载标签和预测图像
    label_path = os.path.join(label_folder, label_file)
    pred_path = os.path.join(pred_folder, pred_file)
    
    # 假设标签和预测都是二值图像
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    
    # 转换为tensor
    label_tensor = torch.tensor(label / 255.0, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
    pred_tensor = torch.tensor(pred / 255.0, dtype=torch.float32).unsqueeze(0)    # [1, H, W]

    label_np = label_tensor.squeeze(0).numpy()

    # 使用cv2将1968x1968放大到2048x2048
    label_resized_np = cv2.resize(label_np, (2048, 2048), interpolation=cv2.INTER_NEAREST)

    # 转回Tensor
    label_resized_tensor = torch.from_numpy(label_resized_np).unsqueeze(0)
    
    # 计算指标
    tp, fp, fn, tn = smp.metrics.get_stats(pred_tensor.long(), label_resized_tensor.long(), mode="binary")
    iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro-imagewise")
    acc = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro-imagewise")
    dice_metric = (2 * iou_score) / (1 + iou_score)
    
    # 累加每个指标的值
    iou_sum += iou_score
    acc_sum += acc
    dice_sum += dice_metric

    # 计算面积
    label_white = np.sum(label == 255)
    pred_white = np.sum(pred == 255)
    
    # 将指标写入 TensorBoard
    writer.add_scalar('Dice', dice_metric, i)
    writer.add_scalar('Accuracy', acc, i)
    writer.add_scalar('IoU', iou_score, i)
    writer.add_scalar('label', label_white, i)
    writer.add_scalar('pred', pred_white, i)
    
    print(f'now: {i}')

# 计算平均值
iou_avg = iou_sum / num_samples
acc_avg = acc_sum / num_samples
dice_avg = dice_sum / num_samples

print(f'Average IoU: {iou_avg}')
print(f'Average Accuracy: {acc_avg}')
print(f'Average Dice: {dice_avg}')

# 关闭 TensorBoard writer
writer.close()
