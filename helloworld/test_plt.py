import matplotlib.pyplot as plt
import torch
import math

dtype= torch.float32
device = torch.device("cpu")

x=torch.linspace(-math.pi,math.pi,2000,device=device,dtype=dtype)
y=torch.sin(x)

a=torch.randn((),device=device,dtype=dtype,requires_grad=True)
b=torch.randn((),device=device,dtype=dtype,requires_grad=True)
c=torch.randn((),device=device,dtype=dtype,requires_grad=True)
d=torch.randn((),device=device,dtype=dtype,requires_grad=True)


fig,ax=plt.subplots()
ax.plot(x,y,'b-',label='sin(x)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_ylim(-2,2)

y_pred = a + b * x + c * x ** 2 + d * x ** 3
line,=ax.plot(x,y_pred.detach(),'r-',label='我没招了')


plt.ioff()

plt.legend()
plt.show()
