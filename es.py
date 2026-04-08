import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 4, 200)
y = np.linspace(-10, 10, 200)
X, Y = np.meshgrid(x, y)

Z = 0.5 * X**2 + 3 * X - Y

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, alpha=0.85)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Superfície: z = 0.5x² + 3x - y")

plt.show()