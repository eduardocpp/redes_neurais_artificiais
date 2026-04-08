import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


N = 100
N_teste = 100
np.random.seed(42) 

g1 = np.random.standard_normal((N, 2)) + [4, 2]
g2 = np.random.standard_normal((N, 2)) + [2, 4]

xc1 = np.concatenate((g1[:, 0], g2[:, 0]), axis=0)
xc2 = np.concatenate((g1[:, 1], g2[:, 1]), axis=0)

Y = np.concatenate((np.zeros(N), np.ones(N))).reshape(-1, 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(g1[:, 0], g1[:, 1], color='blue', label='Grupo 1 (xc1)', s=20)
plt.scatter(g2[:, 0], g2[:, 1], color='red', label='Grupo 2 (xc2)', s=20)
plt.title("Dados Antes do Treinamento")
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")
plt.legend()
plt.grid(True, alpha=0.3)

n_input = 2
n_hidden = 20  
X = np.column_stack((xc1, xc2)) 

W = np.random.randn(n_input, n_hidden)
b = np.random.randn(1, n_hidden)

H = np.tanh(np.dot(X, W) + b)

beta = np.dot(np.linalg.pinv(H), Y)

x_min, x_max = xc1.min() - 1, xc1.max() + 1
y_min, y_max = xc2.min() - 1, xc2.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

grid_points = np.c_[xx.ravel(), yy.ravel()]
H_grid = np.tanh(np.dot(grid_points, W) + b)
Z = np.dot(H_grid, beta)
Z = (Z > 0.5).reshape(xx.shape) 

plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu_r')
plt.scatter(g1[:, 0], g1[:, 1], color='blue', s=20, edgecolors='k')
plt.scatter(g2[:, 0], g2[:, 1], color='red', s=20, edgecolors='k')
plt.title(f"Resultado do Treino (H={n_hidden})")
plt.xlabel("Característica 1")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

Y_pred = (np.dot(H, beta) > 0.5).astype(int)
acuracia = np.mean(Y_pred == Y) * 100
print(f"Treinamento concluído. Acurácia: {acuracia}%")


g1_teste = np.random.standard_normal((N_teste, 2)) + [4, 2]
g2_teste = np.random.standard_normal((N_teste, 2)) + [2, 4] 

X_teste = np.vstack((g1_teste, g2_teste))
Y_teste_real = np.concatenate((np.zeros(N_teste), np.ones(N_teste))).reshape(-1, 1)

H_teste = np.tanh(np.dot(X_teste, W) + b)


Y_pred_cont = np.dot(H_teste, beta)
Y_pred_class = (Y_pred_cont > 0.5).astype(int) 


acuracia_teste = np.mean(Y_pred_class == Y_teste_real) * 100
print(f"Acurácia nos dados de Teste: {acuracia_teste:.2f}%")


plt.figure(figsize=(8, 6))


plt.scatter(g1_teste[:, 0], g1_teste[:, 1], color='blue', marker='^', label='Teste Grupo 1', edgecolors='k')
plt.scatter(g2_teste[:, 0], g2_teste[:, 1], color='red', marker='s', label='Teste Grupo 2', edgecolors='k')


x_min, x_max = X_teste[:,0].min()-1, X_teste[:,0].max()+1
y_min, y_max = X_teste[:,1].min()-1, X_teste[:,1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = (np.dot(np.tanh(np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b), beta) > 0.5).reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.1, cmap='RdBu_r')

plt.title(f"Resultado do Teste - Precisão: {acuracia_teste}%")
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()


# ==========================================
# PLOT APENAS DA SUPERFÍCIE DE CLASSIFICAÇÃO 3D (SEM PONTOS)
# ==========================================

# 1. Recriar a grade (usando resolução fina de 0.05 para o degrau ficar nítido)
# Assumindo que xc1, xc2, W, b, beta já estão definidos no seu script
x_min, x_max = xc1.min() - 1, xc1.max() + 1
y_min, y_max = xc2.min() - 1, xc2.max() + 1
xx_3d, yy_3d = np.meshgrid(np.arange(x_min, x_max, 0.05), 
                           np.arange(y_min, y_max, 0.05))

# 2. Calcular a predição discreta (0 ou 1) para a grade
grid_points_3d = np.c_[xx_3d.ravel(), yy_3d.ravel()]
H_grid_3d = np.tanh(np.dot(grid_points_3d, W) + b)
# Z fica estritamente 0 ou 1
Z_class_3d = (np.dot(H_grid_3d, beta) > 0.5).astype(int).reshape(xx_3d.shape)

# 3. Configurar a figura 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 4. Plotar APENAS a superfície em degrau
# antialiased=False mantém a quina do degrau viva
surf = ax.plot_surface(xx_3d, yy_3d, Z_class_3d, cmap='RdBu_r', 
                       alpha=0.8, linewidth=0, antialiased=False)

# 5. Ajustes estéticos (Labels e Ticks)
ax.set_title('Superfície de Decisão ELM (Z=0 ou Z=1)')
ax.set_xlabel('Característica 1 (xc1)')
ax.set_ylabel('Característica 2 (xc2)')
ax.set_zlabel('Classe Predita')

# Forçar o eixo Z a mostrar apenas os números 0 e 1
ax.set_zticks([0, 1])

# Rotacionar para visualizar bem o "corte"
ax.view_init(elev=30, azim=135) 

# (Opcional) Adicionar barra de cores para referência rápida
# fig.colorbar(surf, shrink=0.5, aspect=10, ticks=[0, 1])

plt.show()