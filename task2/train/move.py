import os
import shutil

# 文件夹路径
folders = [r'C:\Users\HMRda\Desktop\pytorch\task2\train\1', 
           r'C:\Users\HMRda\Desktop\pytorch\task2\train\2', 
           r'C:\Users\HMRda\Desktop\pytorch\task2\train\3', 
           r'C:\Users\HMRda\Desktop\pytorch\task2\train\4', 
           r'C:\Users\HMRda\Desktop\pytorch\task2\train\5',
           r'C:\Users\HMRda\Desktop\pytorch\task2\train\6']

# 目标文件夹路径
destination_folder = r'C:\Users\HMRda\Desktop\pytorch\task2\train\origional'

# 创建目标文件夹（如果不存在）
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 支持的图片文件扩展名
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

# 初始化计数器
counter = 0

# 遍历所有源文件夹并处理图片
for folder in folders:
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        
        # 检查文件是否是图片
        if os.path.isfile(file_path) and os.path.splitext(filename)[1].lower() in image_extensions:
            # 生成新的文件名，格式为000000~000xxx
            new_filename = f"{counter:06d}{os.path.splitext(filename)[1]}"
            new_file_path = os.path.join(destination_folder, new_filename)
            
            # 复制文件到目标文件夹并重命名
            shutil.copy(file_path, new_file_path)
            
            # 更新计数器
            counter += 1

print(f"成功将图片移动并重命名到 {destination_folder} 文件夹下。")
