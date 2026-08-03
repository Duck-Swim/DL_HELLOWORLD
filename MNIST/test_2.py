#广测版
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

# 先重新建一个结构完全相同的模型
net = Net()
# 再把保存的权重塞进去
net.load_state_dict(torch.load('mnist_model.pth', weights_only=True))
net.eval()  # 切到推理模式

transform=transforms.Compose([transforms.Resize((10,10)),transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
val_dataset=datasets.MNIST(root='./data',train=False,transform=transform,download=True)
val_loader=torch.utils.data.DataLoader(val_dataset,batch_size=512)

i=0
correct=0
net.eval()
for images,target in val_loader:
    output=net(images)
    _,pred=output.max(1)
    correct+=(pred==target).sum()
    i+=1

    if i>=10:
        break


accuracy=correct/(512*i)
print(str(accuracy))
