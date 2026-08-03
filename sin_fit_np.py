import numpy as np

x = np.linspace(-np.pi, np.pi, 2000)
y = np.sin(x)

a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

print(f'a={a} b={b} c={c} d={d}')

lr = 1e-6

for i in range(5000):
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    loss = np.square(y - y_pred).sum()

    y_grad = 2.0 * (y - y_pred)
    a_grad = y_grad.sum()
    b_grad = (y_grad * x).sum()
    c_grad = (y_grad * x ** 2).sum()
    d_grad = (y_grad * x ** 3).sum()

    if i % 100 == 0:
        print(str(loss))

    a += a_grad * lr
    b += b_grad * lr
    c += c_grad * lr
    d += d_grad * lr

print(f'a={a:.3f},b={b:.3f},c={c:.3f},d={d:.3f}')