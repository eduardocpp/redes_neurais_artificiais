# Considerandoaaproximaçãopolinomialdasseçõesanteriores, faça:
# •Obtenhaaproximaçõespolinomiaisapartirde10amostrasda funçãogeradora
# fg(x)=1
# 2x2+3x+10somadascomumruídogaussianoN(mean=0,sd=4)
# amostradasentrex=−15ex=10,comumnúmerodeamostrasN=20egrau
# dopolinônimovariandoentrep=1ap=8. Paracadaaproximação,mostreum
# gráficocomafunçãogeradora,asamostraseopolinômioobtido.
# •Responda: OcorreuOverfitting? OcorreuUnderfitting? Emquaiscasosocorreu
# estesfenômenos?
# 4
# • Repita o procedimento para 100 amostras ao invés de 10. Qual o impacto do
# número de amostras na aproximação polinomial?

import numpy as np
import matplotlib.pyplot as plt

def funcao_geradora(X):
    return 0.5*X**2 + 3*X + 10

X = np.random.uniform(-15, 10, 10) #X eh a matriz de entradas, gerada aleatoriamente entre -15 e 10
Y = funcao_geradora(X) + np.random.normal(0, 4, 10) #Y eh a matriz de saida, gerada a partir da funcao_geradora com adicao de ruido gaussiano


X_continuo = np.linspace(-15, 10, 100) #X_continuo eh a matriz de entradas para o grafico, gerada aleatoriamente entre -15 e 10
Y_geradora = funcao_geradora(X_continuo) 

plt.plot(X, Y, 'o', label='Amostras', color='blue')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Amostras geradas a partir da função geradora + Funcao Geradora')
plt.plot(X_continuo, Y_geradora, '-', label='Funcao Geradora', color='green')
plt.legend()
plt.grid()
plt.show()

print(X.shape)
for p in range (1,9):
    H = np.array([[x**i for i in range(0,p+1)]for x in X])
    Hinv = np.linalg.pinv(H)
    W = np.dot(Hinv, Y)
    print("Para p = "+ str(p) +": "+ str(W))
    Yhat = np.dot(H, W)
    print("Polinomio obtido:")
    polinomio = "Y = "
    grau = len(W) - 1
    for i in range (grau, -1, -1):
        w = W[i]
        if i == grau:
            polinomio += f"{w:.4f}X^{i}"
        elif i != 0:
            polinomio += f" + {w:.4f}*X^{i}"
        else:
            polinomio += f" + {w:.4f}"
    print(polinomio)
    plt.plot(X, Y, 'o', label='Amostras', color='blue')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Amostras geradas a partir da função geradora + Funcao Geradora')
    plt.plot(X_continuo, Y_geradora, '-', label='Funcao Geradora', color='green')
    plt.plot(X_continuo, np.dot(np.array([[x**i for i in range(0,p+1)]for x in X_continuo]), W), '-', label='Polinomio de grau '+str(p), color='black')
    plt.legend()
    plt.grid()
    plt.show()



import numpy as np
import matplotlib.pyplot as plt

def funcao_geradora(X):
    return 0.5*X**2 + 3*X + 10

X = np.random.uniform(-15, 10, 100) #X eh a matriz de entradas, gerada aleatoriamente entre -15 e 10
Y = funcao_geradora(X) + np.random.normal(0, 4, 100) #Y eh a matriz de saida, gerada a partir da funcao_geradora com adicao de ruido gaussiano


X_continuo = np.linspace(-15, 10, 100) #X_continuo eh a matriz de entradas para o grafico, gerada aleatoriamente entre -15 e 10
Y_geradora = funcao_geradora(X_continuo) 

plt.plot(X, Y, 'o', label='Amostras', color='blue')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Amostras geradas a partir da função geradora + Funcao Geradora')
plt.plot(X_continuo, Y_geradora, '-', label='Funcao Geradora', color='green')
plt.legend()
plt.grid()
plt.show()

print(X.shape)
for p in range (1,9):
    H = np.array([[x**i for i in range(0,p+1)]for x in X])
    Hinv = np.linalg.pinv(H)
    W = np.dot(Hinv, Y)
    print("Para p = "+ str(p) +": "+ str(W))
    Yhat = np.dot(H, W)
    print("Polinomio obtido:")
    polinomio = "Y = "
    grau = len(W) - 1
    for i in range (grau, -1, -1):
        w = W[i]
        if i == grau:
            polinomio += f"{w:.4f}X^{i}"
        elif i != 0:
            polinomio += f" + {w:.4f}*X^{i}"
        else:
            polinomio += f" + {w:.4f}"
    print(polinomio)
    plt.plot(X, Y, 'o', label='Amostras', color='blue')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Amostras geradas a partir da função geradora + Funcao Geradora')
    plt.plot(X_continuo, Y_geradora, '-', label='Funcao Geradora', color='green')
    plt.plot(X_continuo, np.dot(np.array([[x**i for i in range(0,p+1)]for x in X_continuo]), W), '-', label='Polinomio de grau '+ str(p), color='black')
    plt.legend()
    plt.grid()
    plt.show()