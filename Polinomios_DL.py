import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2 + 2*x -10

x = np.linspace(-100, 100, 10000)
y = f(x)

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Função Quadrática')
plt.grid(True)
plt.show()


def treinar_perceptron(X, y, eta=0.1, max_epochs=100):
    w = np.zeros((X.shape[1], y.shape[1]))
    
    for epoca in range(max_epochs):
        erro_total = 0
        for i in range(X.shape[0]):
            u = np.dot(X[i], w)
            y_pred = np.maximum(0, u)
            erro = y[i] - y_pred
            es = erro * np.where(y_pred >= 0, 1, 0)
            w = w + eta * np.outer(X[i], es) 
            erro_total += 0.5 * erro ** 2
                
        if np.sum(erro_total) == 0:
            break
            
    return w

def prever_perceptron(X, w):
    u = np.dot(X, w)
    return u

X_Entrada = list(zip(x, np.ones(len(x))))
X_Entrada = np.array(X_Entrada)
camada_oculta = np.array([[xi**2, xi, 1] for xi in x])
w = treinar_perceptron(X_Entrada, camada_oculta)
print("Pesos treinados para a camada oculta:", w)
saida_prevista = prever_perceptron(X_Entrada, w)
print("Saída prevista para a camada oculta:", saida_prevista)
print("Camada oculta real:", camada_oculta)