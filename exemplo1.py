N = 10 
n = 2 #n eh o numero de neuronios na camada de entrada
p = 5 #p eh o numero de neuronios na camada oculta
m = 4 #m eh o numero de neuronios na camada de saida

import numpy as np
import matplotlib.pyplot as plt

X = np.random.rand(N,n) #X eh a matriz de entradas
Z = np.random.rand(n,p) #Z eh a matriz de pesos da camada oculta
U = np.dot(X, Z) 
H = np.tanh(U) #H eh a matriz de saida da camada oculta
W = np.random.rand(p,m) #W eh a matriz de pesos da camada de saida
O = np.dot(H, W) #O eh a matriz de saida da rede
Yhat = np.tanh(O) #Yhat eh a matriz de saida da rede apos a funcao de ativacao
Y = np.random.rand(N,m) #Y eh a matriz de saida desejada
E = Yhat - Y #E eh a matriz de erros

print('Conjunto de entradas:', X)
print('Conjunto de labels:', Y)
print('Saida da rede:', Yhat)
print('Erro da rede:', E)
