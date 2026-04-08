import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# ==========================================
# 1. FUNÇÕES DO PERCEPTRON
# ==========================================
def trainperceptron(X, y, eta=0.1, max_epochs=100):
    """Treina os pesos do Perceptron. 
    O max_epochs vai ser o critério de parada principal aqui, 
    já que a base Breast Cancer não é perfeitamente separável linearmente.
    """
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

def prever(X, w):
    u = np.dot(X, w)
    return np.where(u >= 0, 1, 0)

# ==========================================
# 2. CARREGAMENTO E PREPARAÇÃO DOS DADOS BREAST CANCER
# ==========================================
print("Carregando base de dados Breast Cancer...")
cancer_data = load_breast_cancer()
X_cancer = cancer_data.data
y_cancer = cancer_data.target # Já vem em 0 (maligno) e 1 (benigno)

# Adicionando a coluna do Bias (coluna de 1s) nas 30 características da base
bias = np.ones((X_cancer.shape[0], 1))
X_com_bias = np.hstack((bias, X_cancer))


# ==========================================
# 3. LOOP DE 100 TREINAMENTOS
# ==========================================
acuracias_treino = []
acuracias_teste = []

print("Iniciando loop de 100 treinamentos (isso pode levar alguns segundos)...")

for i in range(1):
    # Divisão 70/30 garantindo a proporção de tumores benignos/malignos
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X_com_bias, 
        y_cancer, 
        test_size=0.30, 
        stratify=y_cancer
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
print("=== MATRIZ DE CONFUSÃO (Última iteração do loop) ===")
print("="*50)
print("Treinamento:")
print(confusion_matrix(y_treino, y_pred_treino))
print("\nTeste:")
print(confusion_matrix(y_teste, y_pred_teste))