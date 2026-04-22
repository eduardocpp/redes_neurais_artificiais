import numpy as np
import matplotlib.pyplot as plt

def k_means(X, k, max_epochs = 100):
    plt.scatter(X[:, 0], X[:, 1], c='black', marker='o')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.title('Dados originais')
    plt.show()

    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    for epoch in range(max_epochs):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids
        # plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
        # plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200)
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.grid()
        # plt.title(f'Resultado do K-Means iteracao {epoch+1}')
        # plt.show()
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.title(f'Resultado do K-Means iteracao {epoch+1}')
    plt.show()
    return centroids, labels


def gaussian(x, m, r):
    return (1 / np.sqrt(2 * np.pi * r**2)) * np.exp(-0.5 * ((x - m) / r)**2)


x = np.arange(0, 2*np.pi, 0.1 * np.pi)
y = np.sin(x)
plt.scatter(x, y, label='Seno')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Função Seno')
plt.grid()
plt.legend()
plt.show()

#modelo

h1 = gaussian(x, np.pi/2, 1)
h2 = gaussian(x, 3*np.pi/2, 1)

w1 = 1
w2 = -1
w0 = 0

y_pred = w0 + w1 * h1 + w2 * h2

plt.scatter(x, y, label='Seno', c='blue')   
plt.plot(x, y_pred, label='RBF', linestyle='-', c='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Função Seno e Aproximação RBF')
plt.grid()
plt.legend()
plt.show() 


import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. GERAÇÃO DOS DADOS
# ==========================================
N = 50
np.random.seed(42) 

# Grupos
g11 = np.random.standard_normal((N, 2)) + [6, 2] # Classe 0
g21 = np.random.standard_normal((N, 2)) + [2, 6] # Classe 0
g12 = np.random.standard_normal((N, 2)) + [2, 2] # Classe 1
g22 = np.random.standard_normal((N, 2)) + [6, 6] # Classe 1

# Matriz X (80, 2) e Vetor Y (80,)
X = np.vstack((g11, g21, g12, g22))
# achatando o Y com .ravel() para evitar problemas no plot e nos cálculos
Y = np.concatenate((np.zeros(2*N), np.ones(2*N))).ravel() 

# ==========================================
# 2. FUNÇÃO K-MEANS
# ==========================================
def k_means(X, k, max_epochs=100):
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    for _ in range(max_epochs):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids

# ==========================================
# 3. TREINAMENTO DA REDE RBF
# ==========================================
k = 4 # 4 nuvens de pontos
centros = k_means(X, k)
sigma = 1.5 # Largura da Gaussiana (ajustável)

def rbf_gaussiana(x, c, s):
    return np.exp(-np.linalg.norm(x - c)**2 / (2 * s**2))

# Construindo a Matriz de Ativação G (N, k)
H = np.zeros((X.shape[0], k))
for i in range(X.shape[0]):
    for j in range(k):
        H[i, j] = rbf_gaussiana(X[i], centros[j], sigma)

# Adicionando a coluna de Bias (Tudo 1) à matriz G
H = np.column_stack([np.ones(X.shape[0]), H])

# Calculando os pesos W via Pseudo-Inversa
# W = (G^T * G)^-1 * G^T * Y
W = np.linalg.pinv(H) @ Y

# ==========================================
# 4. PLOTAGEM COM FRONTEIRA DE DECISÃO
# ==========================================
plt.figure(figsize=(12, 5))

# --- Plot 1: Dados Originais Consertados ---
plt.subplot(1, 2, 1)
# Plotamos o X inteiro passando o Y inteiro como cor
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='bwr', edgecolor='k', s=40)
plt.title("Dados Originais (Consertado)")
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")
plt.grid(True, alpha=0.3)

# --- Plot 2: Fronteira de Decisão RBF ---
plt.subplot(1, 2, 2)

# Criando um grid de pontos no gráfico para ver o que a RBF prevê
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), 
                     np.linspace(y_min, y_max, 100))

# Previsão para cada ponto do grid
grid_pontos = np.c_[xx.ravel(), yy.ravel()]
H_grid = np.zeros((grid_pontos.shape[0], k))
for i in range(grid_pontos.shape[0]):
    for j in range(k):
        H_grid[i, j] = rbf_gaussiana(grid_pontos[i], centros[j], sigma)

H_grid = np.column_stack([np.ones(grid_pontos.shape[0]), H_grid])

# Z contém as previsões contínuas. Transformamos em 0 ou 1 com limiar de 0.5
Z_continuo = H_grid @ W
Z = np.where(Z_continuo >= 0.5, 1, 0).reshape(xx.shape)

# Pintando o fundo
plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')

# Desenhando os pontos e os centros por cima
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='bwr', edgecolor='k', s=40)

plt.title("Fronteira de Decisão da RBF")
plt.xlabel("Característica 1")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# 1. Definir os limites e criar a malha (grid)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx_3d, yy_3d = np.meshgrid(np.arange(x_min, x_max, 0.05), 
                           np.arange(y_min, y_max, 0.05))

grid_points_3d = np.c_[xx_3d.ravel(), yy_3d.ravel()]

# ========================================================
# CORREÇÃO 1: Calcular a ativação RBF corretamente 
# (calculando a distância de cada ponto para cada centro)
# ========================================================
H_grid_3d = np.zeros((grid_points_3d.shape[0], k)) # k = 4 centros
for i in range(grid_points_3d.shape[0]):
    for j in range(k):
        H_grid_3d[i, j] = rbf_gaussiana(grid_points_3d[i], centros[j], sigma)

# ========================================================
# CORREÇÃO 2: Adicionar a coluna de Bias (Tudo 1)
# Sem isso, H_grid_3d tem 4 colunas e W tem 5 elementos, dando erro!
# ========================================================
H_grid_3d = np.column_stack([np.ones(grid_points_3d.shape[0]), H_grid_3d])

# 2. Calcular a predição discreta (0 ou 1) para a grade
# Agora o dot product funciona perfeitamente
Z_class_3d = (np.dot(H_grid_3d, W) >= 0.5).astype(int).reshape(xx_3d.shape)

# 3. Configurar a figura 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 4. Plotar APENAS a superfície em degrau
# antialiased=False mantém a quina do degrau viva
surf = ax.plot_surface(xx_3d, yy_3d, Z_class_3d, cmap='RdBu_r', 
                       alpha=0.8, linewidth=0, antialiased=False)

# 5. Ajustes estéticos (Labels e Ticks)
# Alterei "ELM" para "RBF" no título apenas para manter a consistência com o que fizemos
ax.set_title('Superfície de Decisão RBF (Z=0 ou Z=1)') 
ax.set_xlabel('Característica 1 (xc1)')
ax.set_ylabel('Característica 2 (xc2)')
ax.set_zlabel('Classe Predita')

# Forçar o eixo Z a mostrar apenas os números 0 e 1
ax.set_zticks([0, 1])

# Rotacionar para visualizar bem o "corte"
ax.view_init(elev=30, azim=135) 

plt.show()