import torch
import torch.nn as nn
import torch.optim as optim
#直线拟合


net=nn.Linear(1,1)
optimizer=optim.SGD(net.parameters(),lr=0.1)
criterion=nn.MSELoss()
x=torch.randn((1,))
target=torch.tensor([3*x+2])

for i in range(100):
    x = torch.randn((1,))
    target = torch.tensor([3 * x + 2])
    output=net(x)
    loss=criterion(output,target)
    print(round(loss.item(),2))
    print(f'w={net.weight.item()},b={net.bias.item()}')
    net.zero_grad()
    loss.backward()
    optimizer.step()