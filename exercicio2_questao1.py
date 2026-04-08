# 1 Problema Não-Linearmente Separável
# • Objetivos: melhor entendimento do papel das projeções não-lineares na lineariza
# ção de problemas em redes neurais artificiais.
# • Considereoproblema declassificação não-linear da Figura 1, que representa amostras
# duas classes, representadas na figura nas cores vermelha e preta. Um classificador
# capaz de separar amostras das duas classes deve resultar em uma superfície que
# separe as duas regiões. Funções que poderiam resolver este problema são, por ex
# emplo, são funções radiais, circulares ou mesmo aproximações com múltiplas retas
# que resultem na separação das duas regiões. Composições de retas formando uma
# região hexagonal englobando a classe preta, por exemplo, pode resultar em uma
# boa separação. Pede-se que seja implementada, em R, Python ou outra linguagem
# de sua preferência, uma projeção não linear arbitrária que torne o problema lin
# earmente separável. Apresente o gráfico final da superfície de separação. Tente
# trabalhar com uma ou duas funções somente para que o resultado da projeção possa
# ser visualizado. Apresente o gráfico dos pontos projetados neste novo espaço
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-1, 1.1, 0.1) #x eh a matriz de entradas, gerada aleatoriamente entre -1 e 1
y= np.arange(-1, 1.1, 0.1) #y eh a matriz de entradas, gerada aleatoriamente entre -1 e 1

X, Y = np.meshgrid(x, y) #X e Y sao as matrizes de entradas, geradas a partir das matrizes x e y usando a funcao meshgrid

grid = np.column_stack((X.ravel(), Y.ravel())) #grid eh a matriz de entradas, gerada a partir das matrizes X e Y usando a funcao column_stack

plt.scatter(grid[:,0], grid[:,1], c='blue') #grafico dos pontos projetados neste novo espaço
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Grafico dos pontos gerados originalmente sem separacao de classes')
plt.grid()
plt.show()


raio = 0.6
dist = np.sqrt(grid[:,0]**2 + grid[:,1]**2) #dist eh a matriz de distancias, calculada a partir das matrizes x e y usando a funcao sqrt
classe = (dist < raio)

plt.scatter(grid[classe,0], grid[classe,1], c='red') #grafico dos pontos projetados neste novo espaço, com as classes separadas
plt.scatter(grid[~classe,0], grid[~classe,1], c='blue') #grafico dos pontos projetados neste novo espaço, com as classes separadas
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Grafico dos pontos gerados originalmente com separacao de classes')
plt.grid()
plt.show()


# Projecao nao linear usando a funcao de distancia ao centro, que resulta em uma superficie circular de separacao entre as classes

Z = dist**2 #Z eh a matriz de saida da camada oculta, gerada a partir da matriz grid usando a funcao de distancia ao centro

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(grid[classe,0], grid[classe,1], Z[classe], c='red')
ax.scatter(grid[~classe,0], grid[~classe,1], Z[~classe], c='blue')
r2 = raio**2

xx, yy = np.meshgrid(x, y)
zz = np.full_like(xx, r2)

ax.plot_surface(xx, yy, zz, alpha=0.3)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z = X² + Y²')

plt.title("Projecao nao linear dos dados")
plt.show()

