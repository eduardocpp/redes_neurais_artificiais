import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 3*x + 10

x = np.random.uniform(0, 10, 11)
y = f(x) 

w1 = 2.9
w0 = 9.0
yhat = w1*x + w0

erro = (y-yhat)**2

for i in range(len(x)):
    print(f"x: {x[i]:.2f}, y: {y[i]:.2f}, yhat: {yhat[i]:.2f}, erro quadratico: {erro[i]:.2f}, erro absoluto: {abs(y[i]-yhat[i]):.2f}")

w1vec = np.arange(0,6,0.1)
w0vec = np.arange(0,20,0.1)

W1, W0 = np.meshgrid(w1vec, w0vec)

E = np.zeros_like(W1)
print(f"W1 shape: {W1.shape}, W0 shape: {W0.shape}, E shape: {E.shape}")
print(f"W1:{W1}, W0: {W0}, E: {E}")
input()

for i in range(len(W1)):
    for j in range(len(W1[0])):
        w1 = W1[i,j]
        w0 = W0[i,j]
        yhat = w1*x + w0
        E[i,j] = np.sum((y-yhat)**2)
        print(f"E[i,j]: {E[i,j]}, w1: {w1:.2f}, w0: {w0:.2f}")


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(W1, W0, E)

ax.set_xlabel('w1')
ax.set_ylabel('w0')
ax.set_zlabel('Erro')

plt.title('Superficie de erro')
plt.show()