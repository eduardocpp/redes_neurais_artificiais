#ideia do overfitting:

#o risco real de um modelo de rede neural eh limitado superiormente por:
#R <= Rempirico + sqrt(h/N), onde h eh um indicador de complexidade do modelo e N eh o numero de dados do dataset
#Se N eh infinito, temos que R <= Rempirico, entao basta diminuir o Rempirico
#Se N eh fixo, para diminuir o Risco, temos que diminuir h para que a razao h/N diminua, e para que o termo em geral seja reduzido
#Dessa forma, ou aumentamos o numero de dados no dataset para um modelo mais complexo ou simplificamos o modelo para um dataset menor


import numpy as np
import matplotlib.pyplot as plt

def fgx(xin):
    return 0.5*xin**2 + 3*xin + 10

def fgxMaior(xin):
    return 0.25*xin**4 + 0.3*xin**3 + 0.5*xin**2 + 2*xin + 10


#fgx normal, sem overfitting

X = np.random.uniform(-15, 10, 50) #X eh a matriz de entradas, gerada aleatoriamente entre -15 e 10
Y = fgx(X) + 10*np.random.randn(len(X)) #Y eh a matriz de saida desejada, gerada a partir da funcao fgx com um ruido aleatorio


plt.plot(X, Y, 'o')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
plt.show()

H = np.array([[x**2, x, 1] for x in X]) #H eh a matriz de saida da camada oculta, gerada a partir das entradas X
print(X.shape)
print(H.shape)

W = np.dot(np.linalg.pinv(H), Y) #W eh a matriz de pesos da camada de saida, calculada a partir da matriz H e da matriz Y usando a pseudo-inversa

print(W)

Yhat = np.dot(H, W) #Yhat eh a matriz de saida da rede, calculada a partir da matriz H e da matriz W
print(Yhat)
Erro = Yhat - Y #Erro eh a matriz de erros, calculada a partir da matriz Yhat e da matriz Y)

plt.plot(X, Y, 'o', label='Dados', color='blue')
plt.plot(X, Yhat, 'x', label='Previsoes',color='red')
Xcont = np.linspace(-15, 10, 100) #Xcont eh a matriz de entradas para o grafico, gerada aleatoriamente entre -15 e 10
plt.plot(Xcont, fgx(Xcont), '-', label='Erro', color='green')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
plt.show()

#Polinomio de grau maior com overfitting

Y = fgxMaior(X) + 10*np.random.randn(len(X)) #Y eh a matriz de saida desejada, gerada a partir da funcao fgx com um ruido aleatorio


plt.plot(X, Y, 'o')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
plt.show()

H = np.array([[x**4,x**3,x**2, x, 1] for x in X]) #H eh a matriz de saida da camada oculta, gerada a partir das entradas X
print(X.shape)
print(H.shape)

W = np.dot(np.linalg.pinv(H), Y) #W eh a matriz de pesos da camada de saida, calculada a partir da matriz H e da matriz Y usando a pseudo-inversa

print(W)

Yhat = np.dot(H, W) #Yhat eh a matriz de saida da rede, calculada a partir da matriz H e da matriz W
print(Yhat)
Erro = Yhat - Y #Erro eh a matriz de erros, calculada a partir da matriz Yhat e da matriz Y)

plt.plot(X, Y, 'o', label='Dados', color='blue')
plt.plot(X, Yhat, 'x', label='Previsoes',color='red')
Xcont = np.linspace(-15, 10, 100) #Xcont eh a matriz de entradas para o grafico, gerada aleatoriamente entre -15 e 10
plt.plot(Xcont, fgxMaior(Xcont), '-', label='Erro', color='green')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
plt.show()


#Polinomio de grau maior sem overfitting

X = np.random.uniform(-15, 10, 500) #X eh a matriz de entradas, gerada aleatoriamente entre -15 e 10
Y = fgxMaior(X) + 10*np.random.randn(len(X)) #Y eh a matriz de saida desejada, gerada a partir da funcao fgx com um ruido aleatorio


plt.plot(X, Y, 'o')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
plt.show()

H = np.array([[x**2, x, 1] for x in X]) #H eh a matriz de saida da camada oculta, gerada a partir das entradas X
print(X.shape)
print(H.shape)

W = np.dot(np.linalg.pinv(H), Y) #W eh a matriz de pesos da camada de saida, calculada a partir da matriz H e da matriz Y usando a pseudo-inversa

print(W)

Yhat = np.dot(H, W) #Yhat eh a matriz de saida da rede, calculada a partir da matriz H e da matriz W
print(Yhat)
Erro = Yhat - Y #Erro eh a matriz de erros, calculada a partir da matriz Yhat e da matriz Y)

plt.plot(X, Y, 'o', label='Dados', color='blue')
plt.plot(X, Yhat, 'x', label='Previsoes',color='red')
Xcont = np.linspace(-15, 10, 500) #Xcont eh a matriz de entradas para o grafico, gerada aleatoriamente entre -15 e 10
plt.plot(Xcont, fgxMaior(Xcont), '-', label='Erro', color='green')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
plt.show()