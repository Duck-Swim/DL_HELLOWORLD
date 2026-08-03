import torch
import math
import matplotlib.pyplot as plt

dtype= torch.float32
device = torch.device("cpu")

x=torch.linspace(-math.pi,math.pi,2000,device=device,dtype=dtype)
y=torch.sin(x)

a=torch.randn((),device=device,dtype=dtype,requires_grad=True)
b=torch.randn((),device=device,dtype=dtype,requires_grad=True)
c=torch.randn((),device=device,dtype=dtype,requires_grad=True)
d=torch.randn((),device=device,dtype=dtype,requires_grad=True)

print(f'a={a},b={b},c={c},d={d}')

lr=0.000003

fig,ax=plt.subplots()
ax.plot(x,y,'b-',label='sin(x)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_ylim(-2,2)

y_pred = a + b * x + c * x ** 2 + d * x ** 3
line,=ax.plot(x,y_pred.detach(),'r-',label='我没招了')

for i in range(2000):
    y_pred = a + b * x + c * x ** 2 + d * x ** 3
    loss=(y_pred-y).pow(2).sum()

    if i%100==0:
        print(f'loss={loss.item():.4f}')
        line.set_ydata(y_pred.detach())

        plt.pause(0.1)

    loss.backward()

    with torch.no_grad():
        a-=lr*a.grad
        b-=lr*b.grad
        c-=lr*c.grad
        d-=lr*d.grad

        a.grad=None
        b.grad=None
        c.grad=None
        d.grad=None


plt.ioff()

plt.legend()
plt.show()


print(f'{a:.3f}+{b:.3f}x+{c:.3f}x^2+{d:.3f}x^3')