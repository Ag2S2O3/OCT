import os
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# 形态学运算
def opening_closing(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 100))
    
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # 开运算
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  # 闭运算

    return img

def main(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 读取数据
    test_dir = r'C:\Users\HMRda\Desktop\pytorch\OCT\task1\test\img'
    result_dir = r'C:\Users\HMRda\Desktop\pytorch\OCT\task1\test\res'

    # 读取文件夹中的每个文件夹
    folder = os.listdir(test_dir)

    # 读取进来的数据大小转换为512
    PATCH_SIZE = 512
    transform = A.Compose([
        A.Resize(width=PATCH_SIZE, height=PATCH_SIZE),
        ToTensorV2(),
    ])

    # 加载模型
    model = smp.Unet(
        encoder_name='resnet50', in_channels=1, classes=1
    )
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

if __name__ == '__main__':
    main(
        r'C:\Users\HMRda\Desktop\pytorch\OCT\task1\trained\RU_Net_512\checkpoint_best.pth'
    )
