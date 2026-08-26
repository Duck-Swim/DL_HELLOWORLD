import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2., 2., 2000)
y1 = np.sin(x)

a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

a1=a
b1=b
c1=c
d1=d

print(f'a={a} b={b} c={c} d={d}')

lr = 1e-6

for i in range(5000):
    y2 = a + b * x + c * x ** 2 + d * x ** 3
    y3= a1 + b1 * x + c1 * x ** 2 + d1 * x ** 3
    loss1 = np.square(y1 - y2).sum()/2000
    loss2=np.abs(y1-y3).sum()



    y2_grad = 2.0 * (y1 - y2)
    a_grad = y2_grad.sum()
    b_grad = (y2_grad * x).sum()
    c_grad = (y2_grad * x ** 2).sum()
    d_grad = (y2_grad * x ** 3).sum()

    y3_grad = y1 - y2
    a1_grad = y3_grad.sum()
    b1_grad = (y3_grad * x).sum()
    c1_grad = (y3_grad * x ** 2).sum()
    d1_grad = (y3_grad * x ** 3).sum()

    if i % 100 == 0:
        print(f'loss1={loss1}')
        print(f"loss2={loss2}")

    a += a_grad * lr
    b += b_grad * lr
    c += c_grad * lr
    d += d_grad * lr

    a1 += a1_grad * lr
    b1 += b1_grad * lr
    c1 += c1_grad * lr
    d1 += d1_grad * lr

print(f'a={a:.3f},b={b:.3f},c={c:.3f},d={d:.3f}')
print(f'a1={a1:.3f},b1={b1:.3f},c1={c1:.3f},d1={d1:.3f}')



for i in range(5000):
    y4 = a + b * x + c * x ** 2 + d * x ** 3

    y_grad = y1 - y4

    # 权重 = exp(-|y_grad| / sigma)
    sigma = 0.05  # 控制“下降速度”，越大则权重下降越慢
    w = np.exp(-np.abs(y_grad) / sigma)

    # 加权损失
    loss3 = (w * np.square(y_grad)).sum()


    a_grad = (w*y_grad).sum()
    b_grad = (w*y_grad * x).sum()
    c_grad = (w*y_grad * x ** 2).sum()
    d_grad = (w*y_grad * x ** 3).sum()

    if i % 100 == 0:
        print(f'loss1={loss3}')

    a += a_grad * lr
    b += b_grad * lr
    c += c_grad * lr
    d += d_grad * lr

print(f'a={a:.3f},b={b:.3f},c={c:.3f},d={d:.3f}')

fig,ax=plt.subplots(figsize=(10,6))
ax.plot(x,y1,label="sin(x)")
ax.plot(x,y2,label="loss1")
#ax.plot(x,y3,label="loss2")
#ax.plot(x,y4,label="loss3")

ax.legend()
ax.grid(alpha=0.3)
plt.show()


