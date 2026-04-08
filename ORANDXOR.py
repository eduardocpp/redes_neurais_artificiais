import numpy as np
import matplotlib.pyplot as plt

x = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
yor = np.array([0, 1, 1, 1])
yand = np.array([0, 0, 0, 1])
yxor = np.array([0, 1, 1, 0])

def treinar_perceptron(X, y, eta=0.1, max_epochs=6):
    w = np.zeros(X.shape[1]) 
    
    for epoca in range(max_epochs):
        erro_total = 0
        for i in range(X.shape[0]):
            u = np.dot(X[i], w)
            y_pred = 1 if u >= 0 else 0
            
            erro = y[i] - y_pred
            if erro != 0:
                w = w + eta * erro * X[i]
                erro_total += 1
                
        if erro_total == 0:
            break
            
    return w

def prever_perceptron(X, w):
    u = np.dot(X, w)
    return np.where(u >= 0, 1, 0)

def treinar_adaline(X, y, eta=0.01, max_epochs=2000):
    w = np.zeros(X.shape[1]) 
    
    for epoca in range(max_epochs):
        erro_total = 0
        for i in range(X.shape[0]):
            u = np.dot(X[i], w)
            erro = y[i] - u
            
            w = w + eta * erro * X[i]
            erro_total += 0.5 * erro ** 2
            
    return w

def prever_adaline(X, w):
    u = np.dot(X, w)
    return u

w_or = treinar_perceptron(x, yor)
w_and = treinar_perceptron(x, yand)
print("Pesos treinados para OR:", w_or)
print("Pesos treinados para AND:", w_and)

print("Previsões para OR:", prever_perceptron(x, w_or))
print("Previsões para AND:", prever_perceptron(x, w_and))

x_xor = list(zip(yor, yand))
x_xor = np.array(x_xor)
w_xor = treinar_adaline(x_xor, yxor)
print("Pesos treinados para XOR via adaline:", w_xor)
print("Previsões para XOR via adaline:", prever_adaline(x_xor, w_xor))

x_xor = list(zip(yor, yand, np.ones(len(yor))))
x_xor = np.array(x_xor)
w_xor = treinar_perceptron(x_xor, yxor)
print("Pesos treinados para XOR via perceptron:", w_xor)
print("Previsões para XOR via perceptron:", prever_perceptron(x_xor, w_xor))