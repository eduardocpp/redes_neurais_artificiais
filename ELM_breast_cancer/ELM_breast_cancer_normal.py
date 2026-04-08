import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler # 1. Importando o normalizador

# ==========================================
# 1. FUNÇÕES DO ELM
# ==========================================
def trainELM(X, y, n_hidden=400):
    """Treina os pesos do ELM usando a pseudoinversa analítica."""
    n_inputs = X.shape[1]
    beta = np.random.randn(n_inputs, n_hidden)

    H = np.tanh(np.dot(X, beta))
    W = np.dot(np.linalg.pinv(H), y)
            
    return beta, W

def prever(X, beta, W):
    H = np.tanh(np.dot(X, beta))
    u = np.dot(H, W)
    # 2. Limiar ajustado para 0.5 (ponto médio entre as classes 0 e 1)
    return np.where(u >= 0.5, 1, 0)

# ==========================================
# 2. CARREGAMENTO E PREPARAÇÃO DOS DADOS BREAST CANCER
# ==========================================
print("Carregando base de dados Breast Cancer...")
cancer_data = load_breast_cancer()
X_cancer = cancer_data.data
y_cancer = cancer_data.target # 0 (maligno) e 1 (benigno)

print(f"Benignos (1): {len(np.array([y_benigno for y_benigno in y_cancer if y_benigno == 1]))}")
print(f"Malignos (0): {len(np.array([y_maligno for y_maligno in y_cancer if y_maligno == 0]))}")

# IMPORTANTE: Removemos a adição do Bias daqui. Ele irá para dentro do loop.

# ==========================================
# 3. LOOP DE 1000 TREINAMENTOS
# ==========================================
acuracias_treino = []
acuracias_teste = []

maior_acuracia_teste = 0
melhor_beta = None
melhor_W = None
y_teste_melhor_pred = None
y_teste_melhor_real = None # Variável para consertar o erro da matriz de confusão

print("Iniciando loop de 1000 treinamentos (isso pode levar alguns segundos)...")

for i in range(1000):
    # A. Divisão usando os dados originais (sem bias)
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X_cancer, 
        y_cancer, 
        test_size=0.30, 
        stratify=y_cancer
    )
    
    # B. Normalização (Essencial para a ELM não saturar)
    scaler = StandardScaler()
    X_treino_norm = scaler.fit_transform(X_treino)
    X_teste_norm = scaler.transform(X_teste)
    
    # C. Adicionando o Bias APÓS a normalização
    bias_treino = np.ones((X_treino_norm.shape[0], 1))
    X_treino_final = np.hstack((bias_treino, X_treino_norm))
    
    bias_teste = np.ones((X_teste_norm.shape[0], 1))
    X_teste_final = np.hstack((bias_teste, X_teste_norm))
    
    # Treina o modelo passando os dados já com bias
    beta, w_treinado = trainELM(X_treino_final, y_treino, n_hidden=i + 1)
    
    # Faz previsões
    y_pred_treino = prever(X_treino_final, beta, w_treinado)
    y_pred_teste = prever(X_teste_final, beta, w_treinado)
    
    # Calcula e guarda as acurácias
    acuracias_treino.append(accuracy_score(y_treino, y_pred_treino))
    acuracias_teste.append(accuracy_score(y_teste, y_pred_teste))

    # Salva o melhor modelo E o gabarito correspondente a ele
    if(acuracias_teste[-1] > maior_acuracia_teste):
        maior_acuracia_teste = acuracias_teste[-1]
        melhor_beta = beta
        melhor_W = w_treinado
        y_teste_melhor_pred = y_pred_teste
        y_teste_melhor_real = y_teste # Gabarito salvo!

# ==========================================
# 4. RESULTADOS ESTATÍSTICOS
# ==========================================
print("\n" + "="*50)
print("=== RESULTADOS APÓS 1000 ITERAÇÕES ===")
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

print("\n" + "="*50)
print("Melhor Acurácia de Teste Obtida: {:.2f}%".format(maior_acuracia_teste * 100))
print(f"Número de Neurônios na Camada Oculta para Melhor Acurácia: {len(melhor_beta[0])}")
print("Matriz de Confusão para a Melhor Acurácia de Teste:")
# Usando a variável real salva no loop para evitar o ValueError
print(confusion_matrix(y_teste_melhor_real, y_teste_melhor_pred))