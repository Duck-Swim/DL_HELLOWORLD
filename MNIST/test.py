#数据集检查
import torch

from torchvision import datasets,transforms


#数据处理
#1格式压缩，2Numpy转torch，3灰度压缩
transform=transforms.Compose([transforms.Resize((8,8)),transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
#搞来数据集
train_dataset=datasets.MNIST(root='./data',train=True,transform=transform,download=True)
# 加载完数据集后，打印一下长度
#print(len(train_dataset))  # 应该输出 60000

train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=64)

# 从加载器里拿出第一批数据
first_batch = next(iter(train_loader))
print(f"包裹里有 {len(first_batch)} 个东西")   # 应该输出 2
print(f"图像形状: {first_batch[0].shape}")     # 看看尺寸对不对
print(f"标签形状: {first_batch[1].shape}")     # 看看有多少个标签