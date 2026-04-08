import numpy as np

t = np.loadtxt('t', skiprows=1, usecols=1)
x = np.array(np.loadtxt('x', skiprows=1, usecols=(1,2,3)))
y = np.loadtxt('y', skiprows=1, usecols=1)


N = 20

a = np.random.uniform(-10, 10)
W = np.random.uniform(-10,10, size=3)


dataset = list(zip(x,y))
learning_rate = 0.01

for epoch in range(1,200):
    erro_sqr = 0

    for data in dataset:
        Xi = data[0]
        yi = data[1]
        erro = yi - a - np.dot(W, Xi)
        erro_sqr += 0.5 * erro ** 2
        delta_a = -1 * erro
        delta_W = -1 * erro * Xi

        a -= delta_a * learning_rate
        W -= delta_W * learning_rate


print('Modelo y = a + bx1 + cx2 + dx3 co adaline')
print(f'a = {a}')
print(W)
print(f'b = {W[0]}')
print(f'c = {W[1]}')
print(f'd = {W[2]}')


import matplotlib.pyplot as plt

# 1. Calculando a saída prevista para todos os pontos simultaneamente
# O np.dot(x, W) resultará em um vetor 1D do mesmo tamanho de 't' e 'y'
y_previsto = a + np.dot(x, W)

# 2. Montando o Gráfico de Comparação
plt.figure(figsize=(8, 5))

# Plotando os dados originais (linha contínua cinza com bolinhas vazadas, como na Figura 5 )
plt.plot(t, y, marker='o', linestyle='-', color='gray', fillstyle='none', label='Original')

# Plotando a previsão do modelo Adaline (linha tracejada verde com bolinhas preenchidas )
plt.plot(t, y_previsto, marker='o', linestyle='--', color='lightgreen', label='Previsto')

# Configurações visuais do gráfico
plt.title('Avaliação do Modelo Adaline Multivariado')
plt.xlabel('Tempo (t)')
plt.ylabel('Saída (y)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# Exibe o gráfico na tela
plt.show()