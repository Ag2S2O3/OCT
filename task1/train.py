from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2
from OCT_dataset import OCT_Dataset
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import torch.optim as optim
import torch
import epoch as ep
import os
import numpy as np
import random


# 使用命令行管理模型配置
def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="Hello OCT Train", add_help=add_help) # 开启帮助文档
    # 模型和设备配置
    parser.add_argument("--encoder", default="resnet50", type=str, help="encoder, eg: resnet18, mobilenet_v2") 
    parser.add_argument("--model", default="unet", type=str, help="unet/unetpp/deeplabv3p")  
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    # 训练超参数
    parser.add_argument(
        "-b", "--batch-size", default=8, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    ) # 每个 GPU 的 batch 大小，用于控制每次训练中处理的数据量。该大小会影响模型的性能和显存使用
    parser.add_argument("--epochs", default=60, type=int, metavar="N", help="number of total epochs to run") # 总训练轮数
    parser.add_argument(
        "-j", "--workers", default=0, type=int, metavar="N", help="number of data loading workers (default: 1)"
    ) # 定义数据加载时使用的并行线程数，用来加快数据读取的速度
    # 优化器和学习率设置
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer") # 指定优化器类型
    parser.add_argument("--random-seed", default=42, type=int, help="random seed")  
    parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate") # 初始学习率, 决定了每次参数更新的步长
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")    # 用于动量优化器的参数, 帮助在更新过程中平滑梯度
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )   # 权重衰减，用于正则化，防止模型过拟合
    # 学习率调度
    parser.add_argument("--lr-step-size", default=20, type=int, help="decrease lr every step-size epochs") # 每隔指定的 step-size 轮次，降低学习率
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma") # 每次降低学习率时，缩减的倍数。即学习率乘以这个因子（通常小于 1)
    # 输出和恢复训练
    parser.add_argument("--print-freq", default=20, type=int, help="print frequency")
    parser.add_argument("--output-dir", default="./Result", type=str, help="path to save outputs") # 指定输出目录，用于保存训练结果和模型检查点（如模型权重、日志）
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")    # 路径参数，允许从指定的检查点恢复训练
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch") # 如果恢复训练，需要从某个特定的轮次开始
    # 数据增强、优化器附加选项
    parser.add_argument('--autoaug', action='store_true', default=False, help='use torchvision autoaugment') # 启用 AutoAugment，自动化的数据增强策略，可增强模型的泛化能力
    parser.add_argument('--useplateau', action='store_true', default=True, help='use torchvision autoaugment') # 启用 ReduceLROnPlateau 调度器，根据验证损失自动调整学习率
    parser.add_argument('--lowlr', action='store_true', default=False, help='encoder lr divided by 10') # 将编码器的学习率降低 10 倍，通常在微调预训练模型时使用

    return parser

def main(args):
    best_miou = 0.0
    # 选择设备
    device = args.device
    result_dir = args.output_dir
    # 设置日志
    writer = SummaryWriter(log_dir = result_dir)
    # 数据增广
    PATCH_SIZE = 512 # 图像大小
    train_transform = A.Compose([
        A.Resize(width=PATCH_SIZE, height=PATCH_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),
        ToTensorV2(),
    ],additional_targets={'mask': 'mask'})   # 训练集
    valid_transform = A.Compose([
        A.Resize(width=PATCH_SIZE, height=PATCH_SIZE),
        ToTensorV2(),   # 仅数据转换，不会除以255
    ],additional_targets={'mask': 'mask'})  # 验证集

    train_set = OCT_Dataset(r'data\train_data',train_transform)
    valid_set = OCT_Dataset(r'data\valid_data',valid_transform)

    # 构建DataLoader
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    valid_loader = DataLoader(dataset=valid_set, batch_size=args.batch_size, num_workers=args.workers)

    # 构建模型
    model = smp.Unet(
        encoder_name=args.encoder,  # 指定编码器
        encoder_weights='imagenet',   # 使用预训练权重
        in_channels=1,  # 单通道图像
        classes=1,  # 只有一类
        )
    model.to(device)    

    # 选择损失函数
    criterion = smp.losses.DiceLoss(mode='binary')
    
    # 选择优化器
    if args.lowlr: # 如果启用了lowlr
        # encoder 学习率小10倍
        encoder_params_id = list(map(id, model.encoder.parameters()))
        base_params = filter(lambda p: id(p) not in encoder_params_id, model.parameters())
        optimizer = optim.SGD([
            {'params': base_params, 'lr': args.lr},  # 0
            {'params': model.encoder.parameters(), 'lr': args.lr * 0.1}],
            momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)  

    # 设置学习率下降策略
    if args.useplateau: # 如果启用 ReduceLROnPlateau 调度器，根据验证损失自动调整学习率
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               factor=0.1, patience=8, cooldown=2, mode='max')
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size,
                                            gamma=args.lr_gamma)
    
    # 开始训练
    for epoch in range(args.start_epoch, args.epochs):
        print("-----------开始第 {} 轮训练-----------".format(epoch + 1))
        # 训练
        loss_train, miou_train, acc_train = ep.Epoch.train_epoch(
            train_loader, model, device, optimizer, criterion
        )
        # 验证
        loss_valid, miou_valid, acc_valid = ep.Epoch.evaluate_epoch(
            valid_loader, model, device, criterion
        )

        # 获取并更新当前学习率
        lr_current = scheduler.optimizer.param_groups[0]['lr'] if args.useplateau else scheduler.get_last_lr()[0]
        if args.useplateau:
            scheduler.step(miou_valid.avg)
        else:
            scheduler.step()
        
        # 记录
        writer.add_scalars('Loss_group', {'train_loss': loss_train.avg,
                                          'valid_loss': loss_valid.avg}, epoch)
        writer.add_scalars('miou_group', {'train_miou': miou_train.avg,
                                              'valid_miou': miou_valid.avg}, epoch) 
        writer.add_scalar('learning rate', lr_current, epoch)

        # 保存模型
        if best_miou < miou_valid.avg:
            best_epoch = epoch if best_miou < miou_valid.avg else best_epoch
            best_miou = miou_valid.avg if best_miou < miou_valid.avg else best_miou
            pkl_name = "checkpoint_{}.pth".format(epoch) if epoch == args.epochs - 1 else "checkpoint_best.pth"
            path_checkpoint = os.path.join(result_dir, pkl_name)
            torch.save(model.state_dict(), path_checkpoint)
        print("now_IOU = {} , best_epoch = {}".format(miou_valid.avg, best_epoch))

# 设置随机数种子，保证训练一致性
def setup_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)     # cpu
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True       # 训练集变化不大时使训练加速，是固定cudnn最优配置，如卷积算法
    
if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    args = get_args_parser().parse_args()
    setup_seed(args.random_seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args)
    