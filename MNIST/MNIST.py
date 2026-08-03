import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets,transforms
from torch.optim.lr_scheduler import StepLR

#定义神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(1,32,3,1,padding=1)
        self.conv2=nn.Conv2d(32,64,3,1,padding=1)
        self.fc=nn.Linear(1600,10)

    #前向传播计算图（架构）
    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=F.max_pool2d(x,2)
        x=torch.flatten(x,1)
        x=self.fc(x)
        output=F.log_softmax(x,dim=1)
        return output

#三件套：计算图，优化器，标准
net=Net()
optimizer=optim.Adadelta(net.parameters(),lr=1.)
criterion=nn.NLLLoss()

#数据处理
#1格式压缩，2Numpy转torch，3灰度压缩
transform=transforms.Compose([transforms.Resize((10,10)),transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
#搞来数据集
train_dataset=datasets.MNIST(root='./data',train=True,transform=transform,download=True)
#定义数据加载器
train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=512)

val_dataset=datasets.MNIST(root='./data',train=False,transform=transform,download=True)
val_loader=torch.utils.data.DataLoader(val_dataset,batch_size=512)

i=0
#训练循环
for epoch in range(10):
    for images,target in train_loader:
        output=net(images)
        loss=criterion(output,target)
        print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i+=1
        if i>=10:
            print(i)
            i=0
            break


correct=0
net.eval()
for images,target in val_loader:
    output=net(images)
    _,pred=output.max(1)
    correct+=(pred==target).sum()
    i+=1

    if i>=1:
        break


accuracy=correct/(512*i)
print(str(accuracy))

torch.save(net.state_dict(), 'mnist_model.pth')