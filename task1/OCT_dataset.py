import os
import torch
import glob
import numpy as np
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from scipy.ndimage import gaussian_filter

class OCT_Dataset(Dataset):
    # 初始化函数，设定路径，读取路径下的img和mask
    def __init__(self, root_dir, transform=None) -> None:
        super().__init__()
        # 根目录，即存放图像和掩码的主文件夹
        self.root_dir = root_dir
        # 读取图像和掩码的文件路径
        self.img_dir = os.path.join(root_dir, "images")  
        self.mask_dir = os.path.join(root_dir, "masks")    
        # 获取图像和掩码的文件列表并排序（000000-> 00xxxx)
        self.img_fps = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")))  
        self.mask_fps = sorted(glob.glob(os.path.join(self.mask_dir, "*.jpg")))   
        # 检查图像和掩码数量是否一致
        print(f"Number of images: {len(self.img_fps)}")
        print(f"Number of masks: {len(self.mask_fps)}")
        assert len(self.img_fps) == len(self.mask_fps), "图像和掩码的数量不一致！"

        self.transform = transform
        
    def __len__(self):
        if len(self.img_fps) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(
                self.root_dir))  # 代码具有友好的提示功能，便于debug
        return len(self.img_fps)
    
    # 从指定路径下提取img与mask数据
    def __getitem__(self, index):
        img = Image.open(self.img_fps[index]).convert('L')
        mask = Image.open(self.mask_fps[index]).convert('L')

        if self.transform is not None:
            # Convert images to numpy arrays before applying transformations
            img = np.array(img)
            mask = np.array(mask)
            # Apply transformations
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']
        
        # 将mask转换为label
        # 转换为 NumPy 数组
        mask = np.array(mask)

        # 将掩码二值化
        mask = (mask > 0).astype(np.float32)  # 0 表示背景，1 表示前景

        # 转换为 PyTorch 张量  
        '''
        注意这一步是否必要？
        '''
        img = torch.from_numpy(np.array(img)).float()
        mask = torch.from_numpy(mask).float()
    
        return img, mask.long()

# 在此检验是否写对了
if __name__ == "__main__":

    # 数据集的读取

    rootdir = r'data\train_data'

    train_set = OCT_Dataset(root_dir = rootdir)

    train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True)

    for img, mask in train_loader:
        print(f"Image shape: {img.shape}")
        print(f"Mask shape: {mask.shape}")

        img_np = img.squeeze().numpy()  # (H, W)
        mask_np = mask.squeeze().numpy()  # (H, W)

        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.title('Image')
        plt.imshow(img_np, cmap='gray')
        # plt.colorbar()  # 显示颜色条以指示像素值

        plt.subplot(1, 2, 2)
        plt.title('Mask')
        plt.imshow(mask_np, cmap='gray')
        # plt.colorbar()  # 显示颜色条以指示像素值

        plt.show()
        break

    # 验证集的读取

    rootdir = r'data\vaild_data'

    train_set = OCT_Dataset(root_dir = rootdir)

    train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True)

    for img, mask in train_loader:
        print(f"Image shape: {img.shape}")
        print(f"Mask shape: {mask.shape}")

        img_np = img.squeeze().numpy()  # (H, W)
        mask_np = mask.squeeze().numpy()  # (H, W)

        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.title('Image')
        plt.imshow(img_np, cmap='gray')

        plt.subplot(1, 2, 2)
        plt.title('Mask')
        plt.imshow(mask_np, cmap='gray')

        plt.show()
        break