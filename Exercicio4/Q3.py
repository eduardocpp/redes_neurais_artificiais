import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# ==========================================
# 1. FUNÇÕES DO PERCEPTRON (Copiadas do Ex 2)
# ==========================================
def trainperceptron(X, y, eta=0.1, max_epochs=20):
    """Treina os pesos do Perceptron. Aumentei max_epochs por segurança em dados reais."""
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
            
    return w

def prever(X, w):
    u = np.dot(X, w)
    return np.where(u >= 0, 1, 0)

# ==========================================
# 2. CARREGAMENTO E PREPARAÇÃO DOS DADOS IRIS
# ==========================================
print("Carregando base de dados Iris...")
iris = load_iris()
X_iris = iris.data
y_iris_original = iris.target

# Adicionando a coluna do Bias (coluna de 1s) nas 4 dimensões originais
bias = np.ones((X_iris.shape[0], 1))
X_com_bias = np.hstack((bias, X_iris))

# Ajustando os rótulos conforme o PDF: 
# Espécie 1 (target 0) vira Classe 0. Espécies 2 e 3 (targets 1 e 2) viram Classe 1.
y_binario = np.where(y_iris_original == 0, 0, 1)


# ==========================================
# 3. LOOP DE 100 TREINAMENTOS
# ==========================================
# Listas para guardar a acurácia de cada uma das 100 rodadas
acuracias_treino = []
acuracias_teste = []

print("Iniciando loop de 100 treinamentos...")

for i in range(100):
    # O parâmetro stratify=y_binario garante os 70/30 para CADA classe
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X_com_bias, 
        y_binario, 
        test_size=0.30, 
        stratify=y_binario
    )
    
    # Treina o modelo
    w_treinado = trainperceptron(X_treino, y_treino)
    
    # Faz previsões
    y_pred_treino = prever(X_treino, w_treinado)
    y_pred_teste = prever(X_teste, w_treinado)
    
    # Calcula e guarda as acurácias
    acuracias_treino.append(accuracy_score(y_treino, y_pred_treino))
    acuracias_teste.append(accuracy_score(y_teste, y_pred_teste))

# ==========================================
# 4. RESULTADOS ESTATÍSTICOS
# ==========================================
print("\n" + "="*50)
print("=== RESULTADOS APÓS 100 ITERAÇÕES ===")
print("="*50)

print("CONJUNTO DE TREINAMENTO:")
print(f"Acurácia Média: {np.mean(acuracias_treino) * 100:.2f}%")
print(f"Variância:      {np.var(acuracias_treino):.6f}")

print("\nCONJUNTO DE TESTE:")
print(f"Acurácia Média: {np.mean(acuracias_teste) * 100:.2f}%")
print(f"Variância:      {np.var(acuracias_teste):.6f}")

print("\n" + "="*50)
print("=== MATRIZ DE CONFUSÃO (Última iteração do loop) \n [[TN, FP],\n[FN, TP] ===")
print("="*50)
print("Treinamento:")
print(confusion_matrix(y_treino, y_pred_treino))
print("\nTeste:")
print(confusion_matrix(y_teste, y_pred_teste))