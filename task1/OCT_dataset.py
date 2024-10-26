import os
import torch
import glob
import numpy as np
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from scipy.ndimage import gaussian_filter
from albumentations.pytorch import ToTensorV2
import albumentations as A

class OCT_Dataset(Dataset):
    # 初始化函数，设定路径，读取路径下的img和mask
    def __init__(self, root_dir, transform=None, sequence_length=3) -> None:
        super().__init__()
        # 根目录，即存放图像和掩码的主文件夹
        self.root_dir = root_dir
        # 读取图像和掩码的文件路径
        self.img_root_dir = os.path.join(root_dir, "images")  
        self.mask_root_dir = os.path.join(root_dir, "masks")  # masks 也分组
        # 获取 images 目录下的所有子文件夹
        self.img_subdirs = sorted([d for d in os.listdir(self.img_root_dir) if os.path.isdir(os.path.join(self.img_root_dir, d))])

         # 存储每组图像和对应的掩码路径
        self.groups = []
        for subdir in self.img_subdirs:
            img_dir = os.path.join(self.img_root_dir, subdir)
            mask_dir = os.path.join(self.mask_root_dir, subdir)  # 掩码与图像路径对应的子文件夹

            self.img_fps = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
            self.mask_fps = sorted(glob.glob(os.path.join(mask_dir, "*.jpg")))

            print(f"Number of images: {len(self.img_fps)}")
            print(f"Number of masks: {len(self.mask_fps)}")

            # 检查图像和掩码数量是否一致
            assert len(self.img_fps) == len(self.mask_fps), "图像和掩码的数量不一致！"

            self.groups.append((self.img_fps, self.mask_fps))
        
        self.sequence_length = sequence_length

        self.transform = transform
        
    def __len__(self):
        total_sequences = 0
        for img_fps, _ in self.groups:  # 遍历每组数据
            if len(img_fps) == 0:
                raise Exception(f"\nGroup contains no images! Please check your path to images in group.")  # 代码具有友好的提示功能，便于debug
            total_sequences += len(img_fps) - self.sequence_length + 1  # 计算每组能生成的序列数
        return total_sequences

    
    # 从指定路径下提取img与mask数据
    def __getitem__(self, index):

        group_index = 0
        offset = index

        # 确定 index 属于哪一组
        for img_fps, mask_fps in self.groups:
            group_length = len(img_fps) - self.sequence_length + 1
            if offset < group_length:
                break
            offset -= group_length
            group_index += 1

        img_fps, mask_fps = self.groups[group_index]

        # 加载图像序列
        img_sequence = []
        for i in range(self.sequence_length):
            img = Image.open(img_fps[offset + i]).convert('L')
            if self.transform:
                img = np.array(img)
                transformed = self.transform(image=img)
                img = transformed['image']
            img_sequence.append(img)

        # 加载掩码
        mask = Image.open(mask_fps[offset + self.sequence_length - 1]).convert('L')
        if self.transform:
            mask = np.array(mask)
            transformed = self.transform(image=mask)
            mask = transformed['image']
        
        # 将mask转换为label
        # 转换为 NumPy 数组
        mask = np.array(mask)

        # 将掩码二值化
        mask = (mask > 0).astype(np.float32)  # 0 表示背景，1 表示前景

        # 转换为 PyTorch 张量  
        img_sequence = torch.stack([torch.from_numpy(np.array(img)).float() for img in img_sequence])
        mask = torch.from_numpy(mask).float()
        
        return img_sequence, mask.long()

# 在此检验是否写对了
if __name__ == "__main__":

    sequence_length = 10

    train_transform = A.Compose([
        A.Resize(width=512, height=512),
        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
        # A.RandomRotate90(p=0.5),
        # A.Transpose(p=0.5),
        # A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),
        ToTensorV2(),
    ],additional_targets={'mask': 'mask'})   # 训练集

    # 数据集的读取

    rootdir = r'C:\Users\HMRda\Desktop\pytorch\OCT\task1\data\train_data'

    train_set = OCT_Dataset(root_dir = rootdir, sequence_length=10, transform = train_transform)

    train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True)

    print(f"Total number of sequences: {len(train_loader)}")

    for img_sequence, mask in train_loader:
        # 可视化图像序列的第一帧
        plt.figure(figsize=(12, 6))
        for i in range(sequence_length):
            img_sequence= img_sequence.squeeze()
            plt.subplot(2, sequence_length, i + 1)
            plt.imshow(img_sequence[i].numpy(), cmap='gray')
            plt.title(f'Image {i+1}')
            plt.axis('off')

        # 可视化掩码
        plt.subplot(2, sequence_length, sequence_length + 1)
        mask= mask.squeeze()
        plt.imshow(mask.numpy(), cmap='gray')
        plt.title('Mask')
        plt.axis('off')

        plt.show()

        break

    # # 验证集的读取

    # rootdir = r'C:\Users\HMRda\Desktop\pytorch\OCT\task1\data\valid_data'

    # train_set = OCT_Dataset(root_dir = rootdir)

    # train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True)

    # for img, mask ,lasto in train_loader:
    #     print(f"Image shape: {img.shape}")
    #     print(f"Mask shape: {mask.shape}")

    #     img_np = img.squeeze().numpy()  # (H, W)
    #     mask_np = mask.squeeze().numpy()  # (H, W)
    #     lasto_np = lasto.squeeze().numpy()  # (H, W)

    #     plt.figure(figsize=(10, 5))
        
    #     plt.subplot(1, 3, 1)
    #     plt.title('Image')
    #     plt.imshow(img_np, cmap='gray')
    #     # plt.colorbar()  # 显示颜色条以指示像素值

    #     plt.subplot(1, 3, 2)
    #     plt.title('Mask')
    #     plt.imshow(mask_np, cmap='gray')
    #     # plt.colorbar()  # 显示颜色条以指示像素值

    #     plt.subplot(1, 3, 3)
    #     plt.title('last')
    #     plt.imshow(lasto_np, cmap='gray')
    #     # plt.colorbar()  # 显示颜色条以指示像素值
    #     plt.show()
    #     break