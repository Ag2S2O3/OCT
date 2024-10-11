import torch
import numpy as np
import segmentation_models_pytorch as smp


# 每一步所需函数
class Epoch(object):

    # 训练一轮
    @staticmethod
    def train_epoch(dataloader, model, device, optimizer, loss_fn):
        
        # 参数声明
        miou_m = AverageMeter() # 交并比（IoU）的平均值
        loss_m = AverageMeter() # loss的平均值
        acc_m = AverageMeter()  # 准确率的平均值

        step = 0

        # 开始训练
        model.train()   # 将模型设置为训练模式
        for _, data in enumerate(dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # 使用优化器
            optimizer.zero_grad()
            loss = loss_fn(outputs.cpu().squeeze(), labels.cpu())
            loss.backward()
            optimizer.step()

            # 计算指标（miou）
            outputs = (outputs.sigmoid() > 0.5).float() 
            labels = labels.unsqueeze(dim=1)
            tp, fp, fn, tn = smp.metrics.get_stats(outputs.long(), labels, mode="binary")
            iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro-imagewise")  # 由于大量阴性图片的存在，因此macro-imagewise的iou要高，且合理
            acc = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro-imagewise")

            # 记录指标
            loss_m.update(loss.item(), inputs.size(0))  # 因update里： self.sum += val * n， 因此需要传入batch数量
            miou_m.update(iou_score.item(), outputs.size(0))
            acc_m.update(acc.item(), outputs.size(0))

            step = step + 1
            # if step % 100 == 0:
            print("已训练个数: {}".format(step))

        # 返回loss值、IOU和acc
        return loss_m, miou_m, acc_m
    
    # 评估一轮
    @staticmethod
    def evaluate_epoch(dataloader, model, device, loss_fn):
        
        # 参数声明
        miou_m = AverageMeter() # 交并比（IoU）的平均值
        loss_m = AverageMeter() # loss的平均值
        acc_m = AverageMeter()  # 准确率的平均值

        step = 0
        
        # 开始评估
        model.eval()
        with torch.no_grad():  # 不计算梯度
            for _, data in enumerate(dataloader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs.cpu().squeeze(), labels.cpu())

                # 计算指标（miou）
                outputs = (outputs.sigmoid() > 0.5).float() 

                labels = labels.unsqueeze(dim=1)
                tp, fp, fn, tn = smp.metrics.get_stats(outputs.long(), labels, mode="binary")
                iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro-imagewise")  # 由于大量阴性图片的存在，因此macro-imagewise的iou要高，且合理
                acc = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro-imagewise")

                # 记录指标
                loss_m.update(loss.item(), inputs.size(0))  # 因update里： self.sum += val * n， 因此需要传入batch数量
                miou_m.update(iou_score.item(), outputs.size(0))
                acc_m.update(acc.item(), outputs.size(0))

                step = step + 1
                # if step % 100 == 0:
                print("已验证个数: {}".format(step))

        # 返回loss值、IOU和acc
        return loss_m, miou_m, acc_m
     

# 计算平均值的类
class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count