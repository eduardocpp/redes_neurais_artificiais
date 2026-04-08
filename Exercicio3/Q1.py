import numpy as np

t = np.loadtxt('Ex1_t', skiprows=1, usecols=1)
x = np.loadtxt('Ex1_x', skiprows=1, usecols=1)
y = np.loadtxt('Ex1_y', skiprows=1, usecols=1)

N = 20
somaXi = np.sum(x)
somaYi = np.sum(y)
somaXiYi = np.sum(x*y)
somaXi2 = np.sum(x**2)

b = (N * somaXiYi - somaXi * somaYi) / (N * somaXi2 - (somaXi)**2)
a = (somaYi - b * somaXi) / N

print('Modelo a + bx via formula normal de regressao linear')
print(f'a = {a}')
print(f'b = {b}')


a = np.random.uniform(-10, 10)
b = np.random.uniform(-10, 10)

learning_rate = 0.01

dataset = list(zip(x,y))


for epoch in range (1, 200):
    erro_sqr = 0.0

    for data in dataset:
        xi = data[0]
        yi = data[1]
        erro = yi - b * xi - a
        erro_sqr = 0.5 * erro ** 2
        
        delta_b =  -1 * erro * xi
        delta_a = -1 * erro

        b -= learning_rate * delta_b
        a -= learning_rate * delta_a


print('Modelo y = a + bx via treinamento adaline')
print(f'a = {a}')
print(f'b = {b}')


import matplotlib.pyplot as plt

# 1. Gerar novos dados de tempo com um intervalo de amostragem menor
# Usamos o np.linspace para criar 100 pontos entre o tempo mínimo e máximo original
t_teste = np.linspace(min(t), max(t), 100) 

# Gerar os novos dados de entrada senoidal baseados nesse novo tempo
x_teste = np.sin(t_teste) 

# 2. Avaliar a resposta (Calcular a saída usando o modelo Adaline treinado)
y_previsto = a + b * x_teste

# 3. Criar a visualização do teste (semelhante à Figura 2)
plt.figure(figsize=(8, 5))

# Plotando os dados originais (bolinhas vazias, como no documento)
# t e y são as variáveis que você carregou dos arquivos Ex1_t e Ex1_y
plt.plot(t, y, marker='o', linestyle='', color='gray', fillstyle='none', label='Original (Treinamento)')

# Plotando a previsão do Adaline para os novos dados (bolinhas preenchidas verdes)
plt.plot(t_teste, y_previsto, marker='o', linestyle='--', color='lightgreen', label='Previsto (Teste)', markersize=4)

# Configurações do gráfico
plt.title('Avaliação Visual do Modelo Adaline (Teste)')
plt.xlabel('x (Tempo)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.show()