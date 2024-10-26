import os
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import TransferLearning as TrL

# 形态学运算
def opening_closing(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 100))
    
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # 开运算
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  # 闭运算

    return img


# 单输入模型输出
def main(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 读取数据
    test_dir = r'C:\Users\HMRda\Desktop\pytorch\OCT\task1\test\img'
    result_dir = r'C:\Users\HMRda\Desktop\pytorch\OCT\task1\test\res'

    # 读取文件夹中的每个文件夹
    folder = os.listdir(test_dir)

    # 读取进来的数据大小转换为256
    PATCH_SIZE = 256
    transform = A.Compose([
        A.Resize(width=PATCH_SIZE, height=PATCH_SIZE),
        ToTensorV2(),
    ])

    # 加载模型
    model = TrL.TRUnet(hidden_dim=64)
    state_dict = torch.load(model_path, map_location=device)
    model_state_dict = state_dict['model_state_dict']
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    print("启动!")

    # 处理每个图像文件
    for folder in folder:
        test_folder = os.path.join(test_dir,folder)
        result_folder = os.path.join(result_dir,folder)
        for img_name in tqdm(os.listdir(test_folder)):
            img_path = os.path.join(test_folder, img_name)
            if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            with torch.no_grad():
                img = Image.open(img_path).convert('L')
                img = np.array(img)
                transformed = transform(image=img)
                img = transformed['image']
                img_tensor = img.unsqueeze(0).to(device).float()

                # 推理
                output = model(img_tensor)

                # 提取结果
                output = (output.sigmoid() > 0.5).float()
                output = output.squeeze().cpu().numpy().astype('uint8')
                output = cv2.resize(output, (2048, 2048), interpolation=cv2.INTER_NEAREST)
                output = cv2.GaussianBlur(output, (25, 25), 0)
                output[output > 0.5] = 255
                output[output < 0.5] = 0

                # 进行形态学运算
                output = opening_closing(output)

                if not os.path.exists(result_folder):
                    os.makedirs(result_folder)

                # 生成保存路径
                result_path = os.path.join(result_folder, os.path.splitext(img_name)[0] + '.jpg')
                cv2.imwrite(result_path, output)

    print("完毕！")

# 时间序列模型输出
def inference(model_path):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 读取数据
    test_dir = r'C:\Users\HMRda\Desktop\pytorch\OCT\task1\test\img'
    result_dir = r'C:\Users\HMRda\Desktop\pytorch\OCT\task1\test\res\TRUnet-3p'

    # 读取文件夹中的每个文件夹
    folder = os.listdir(test_dir)

    # 读取进来的数据大小转换为256
    PATCH_SIZE = 256
    transform = A.Compose([
        A.Resize(width=PATCH_SIZE, height=PATCH_SIZE),
        ToTensorV2(),
    ])


    # 时间序列长度
    sequence_length = 3
    img_sequence = []

    # 加载模型
    model = TrL.TRUnet(hidden_dim=64)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 创建结果文件夹（如果不存在）
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 遍历每个文件夹（假设文件夹名称代表不同的数据组）
    for folder in os.listdir(test_dir):
        test_folder = os.path.join(test_dir, folder)
        result_folder = os.path.join(result_dir, folder)
        
        # 获取文件夹中的图片文件列表
        img_files = sorted([f for f in os.listdir(test_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # 确保文件数量足够形成序列
        if len(img_files) < sequence_length:
            print(f"Skipping folder {folder}, not enough images for a sequence.")
            continue
        
        # 遍历图片文件并生成序列
        for index in tqdm(range(len(img_files) - sequence_length + 1)):
            img_sequence = []
            
            # 读取图片序列
            for i in range(sequence_length):
                img_path = os.path.join(test_folder, img_files[index + i])
                img = Image.open(img_path).convert('L')  # 转为灰度
                if transform:
                    img = np.array(img)
                    transformed = transform(image=img)  # 应用与训练相同的变换
                    img = transformed['image']          # 增加通道维度
                img_sequence.append(img)
            
            # 将图片序列转换为模型输入所需的格式
            img_sequence = torch.stack(img_sequence, dim=0)  # 转为 (sequence_length, 1, H, W)
            img_sequence = img_sequence.unsqueeze(0).to(device).float()  # 增加 batch 维度，形状 (1, sequence_length, 1, H, W)
            img_sequence.to(device)

            with torch.no_grad():
                # 模型推理
                output = model(img_sequence)

                 # 提取结果
                output = (output.sigmoid() > 0.5).float()
                output = output.squeeze().cpu().numpy().astype('uint8')
                output = cv2.resize(output, (2048, 2048), interpolation=cv2.INTER_NEAREST)
                output = cv2.GaussianBlur(output, (25, 25), 0)
                output[output > 0.5] = 255
                output[output < 0.5] = 0

                 # 进行形态学运算
                output = opening_closing(output)

                if not os.path.exists(result_folder):
                    os.makedirs(result_folder)

                img_name = img_files[index + sequence_length - 1]
                result_path = os.path.join(result_folder, os.path.splitext(img_name)[0] + '.jpg')
                cv2.imwrite(result_path, output)
    

if __name__ == '__main__':

    inference(r'C:\Users\HMRda\Desktop\pytorch\OCT\task1\trained\TRUnet_3p\checkpoint_best.pth')
