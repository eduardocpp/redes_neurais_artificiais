import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Definindo os parâmetros
n_amostras = 200

covariancia = [[0.16, 0], 
               [0, 0.16]]

# 2. Gerando os dados da Classe 1
media_1 = [2, 2]
X1 = np.random.multivariate_normal(media_1, covariancia, n_amostras)

# 3. Gerando os dados da Classe 2
media_2 = [4, 4]
X2 = np.random.multivariate_normal(media_2, covariancia, n_amostras)

# 4. Visualizando os dados (Plot)
plt.figure(figsize=(8, 6))

# Plotando com círculos vazados para ficar parecido com o R
plt.scatter(X1[:, 0], X1[:, 1], facecolors='none', edgecolors='red', label='Classe 1 (Média 2,2)')
plt.scatter(X2[:, 0], X2[:, 1], facecolors='none', edgecolors='blue', label='Classe 2 (Média 4,4)')

# Ajustes do gráfico para bater com os eixos do exercício original
plt.xlim(-2, 8)
plt.ylim(-2, 8)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.title('Dados Amostrados - Distribuições Normais')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


# --- 1. Preparando os dados ---

# Juntando as matrizes X1 e X2 em um único conjunto de dados
X_dados = np.vstack((X1, X2))

# Adicionando a coluna do Bias (coluna de 1s no início)
bias = np.ones((X_dados.shape[0], 1))
X_com_bias = np.hstack((bias, X_dados))

# Criando os rótulos (y): 200 zeros para a Classe 1, 200 uns para a Classe 2
y_dados = np.hstack((np.zeros(n_amostras), np.ones(n_amostras)))

# Embaralhando os dados (boa prática em IA para o modelo não "decorar" a ordem)
indices = np.arange(X_com_bias.shape[0])
np.random.shuffle(indices)
X_treino = X_com_bias[indices]
y_treino = y_dados[indices]


# --- 2. Implementando o Perceptron ---

def trainperceptron(X, y, eta=0.1, max_epochs=100):
    """
    Treina um Perceptron Simples.
    eta: Taxa de aprendizado (learning rate)
    max_epochs: Limite máximo de iterações sobre o dataset
    """
    n_amostras, n_features = X.shape
    
    # Inicializando o vetor de pesos (w) com zeros
    # Como adicionamos o bias no X, w terá tamanho 3: [w0, w1, w2]
    w = np.zeros(n_features)
    
    for epoca in range(max_epochs):
        erro_total = 0
        
        for i in range(n_amostras):
            # Passo A: Calcula a soma ponderada (u = X * w)
            u = np.dot(X[i], w)
            
            # Passo B: Aplica a função de ativação (Degrau)
            y_pred = 1 if u >= 0 else 0
            
            # Passo C: Calcula o erro
            erro = y[i] - y_pred
            
            # Passo D: Atualiza os pesos (Regra de Hebb/Perceptron)
            if erro != 0:
                w = w + eta * erro * X[i]
                erro_total += 1
            
    return w

# --- 3. Executando o treinamento ---

w_final = trainperceptron(X_treino, y_treino)
print("Vetor de pesos encontrado (w):", w_final)


# --- 1. Preparação para os Gráficos ---
# Cria uma figura com espaço para dois gráficos (1 linha, 2 colunas)
fig = plt.figure(figsize=(14, 6))

# --- 2. Gráfico 2D: Dispersão e Fronteira de Decisão ---
ax1 = fig.add_subplot(1, 2, 1)

# Plotando os dados originais
ax1.scatter(X1[:, 0], X1[:, 1], facecolors='none', edgecolors='red', label='Classe 1')
ax1.scatter(X2[:, 0], X2[:, 1], facecolors='none', edgecolors='blue', label='Classe 2')

# Criando a reta de decisão baseada nos pesos (w_final) que você treinou
x1_reta = np.linspace(0, 6, 100)
# Aplicação da fórmula: x2 = -(w0 + w1*x1) / w2
x2_reta = -(w_final[0] + w_final[1] * x1_reta) / w_final[2]

ax1.plot(x1_reta, x2_reta, color='black', linewidth=3, label='Fronteira')
ax1.set_xlim(0, 6)
ax1.set_ylim(0, 6)
ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.legend()
ax1.set_title('Classificação 2D')
ax1.grid(True, linestyle='--', alpha=0.6)


# --- 3. Gráfico 3D: Superfície de Resposta ---
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

# Criando uma malha de coordenadas (grid) de 0 a 6 com passo de 0.1
seqi = np.arange(0, 6.1, 0.1)
seqj = np.arange(0, 6.1, 0.1)
X1_mesh, X2_mesh = np.meshgrid(seqi, seqj)

# Calculando a ativação para todos os pontos da malha simultaneamente
U = w_final[0] + w_final[1] * X1_mesh + w_final[2] * X2_mesh

# Aplicando a função degrau: se U >= 0, recebe 1 (azul), senão 0 (vermelho)
M = np.where(U >= 0, 1, 0)

# Plotando a superfície (cmap='jet' gera o padrão de cores do azul pro vermelho)
surf = ax2.plot_surface(X1_mesh, X2_mesh, M, cmap='jet_r', edgecolor='none', alpha=0.9)

ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.set_zlabel('Saída do Perceptron')
ax2.set_title('Superfície de Decisão 3D')

# Ajustando o ângulo de visão da câmera 3D para ficar parecido com o PDF
ax2.view_init(elev=35, azim=-125)

# Adicionando a barra de cores lateral
fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=10)

plt.tight_layout()
plt.show()