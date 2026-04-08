import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# ==========================================
# 1. GERAÇÃO DOS DADOS (Conforme Exercício 1 e 2)
# ==========================================
n_amostras = 200 # 200 amostras da classe 1 e 200 da classe 2 
# Usando a variância ajustada de 0.16 para manter os dados separáveis
covariancia = [[0.16, 0], 
               [0, 0.16]]

X1 = np.random.multivariate_normal([2, 2], covariancia, n_amostras)
X2 = np.random.multivariate_normal([4, 4], covariancia, n_amostras)

# Juntando os dados e adicionando o Bias (coluna de 1s)
X_dados = np.vstack((X1, X2))
bias = np.ones((X_dados.shape[0], 1))
X_com_bias = np.hstack((bias, X_dados))

# Criando os rótulos: 0 para Classe 1, 1 para Classe 2
y_dados = np.hstack((np.zeros(n_amostras), np.ones(n_amostras)))


# ==========================================
# 2. DIVISÃO EM TREINO E TESTE
# ==========================================
# Separando 70% para treinamento e 30% para teste de forma aleatória 
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X_com_bias, 
    y_dados, 
    test_size=0.30, 
    random_state=1 # Mantém a aleatoriedade reprodutível
)


# ==========================================
# 3. ALGORITMO DO PERCEPTRON
# ==========================================
def trainperceptron(X, y, eta=0.1, max_epochs=100):
    """Treina os pesos do Perceptron"""
    w = np.zeros(X.shape[1]) # Inicializa pesos com zero
    
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

def prever(X, w):
    """Aplica os pesos treinados para fazer previsões"""
    u = np.dot(X, w)
    return np.where(u >= 0, 1, 0)


# ==========================================
# 4. TREINAMENTO E AVALIAÇÃO
# ==========================================
# Usando o conjunto de treinamento para encontrar os pesos 
w_treinado = trainperceptron(X_treino, y_treino)

# Avaliando o desempenho nos dois conjuntos 
y_pred_treino = prever(X_treino, w_treinado)
y_pred_teste = prever(X_teste, w_treinado)


# Apresentando a acurácia e a matriz de confusão 
print("=========================================")
print("=== RESULTADOS: CONJUNTO DE TREINAMENTO ===")
print("=========================================")
print(f"Acurácia: {accuracy_score(y_treino, y_pred_treino) * 100:.2f}%")
print("Matriz de Confusão:")
print(confusion_matrix(y_treino, y_pred_treino))

print("\n=========================================")
print("=== RESULTADOS: CONJUNTO DE TESTE ===")
print("=========================================")
print(f"Acurácia: {accuracy_score(y_teste, y_pred_teste) * 100:.2f}%")
print("Matriz de Confusão:")
print(confusion_matrix(y_teste, y_pred_teste))