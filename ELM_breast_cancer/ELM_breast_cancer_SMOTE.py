import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# ==========================================
# 1. FUNÇÕES DA ELM (Ajustadas para o exemplo)
# ==========================================
def treinar_elm(X, y, n_oculta=100):
    """Treina uma ELM usando a Pseudoinversa"""
    n_amostras, n_features = X.shape
    
    # Pesos aleatórios fixos (Entrada -> Oculta)
    W = np.random.randn(n_features, n_oculta)
    b = np.random.randn(1, n_oculta)
    
    # Matriz H (usando ReLU para evitar saturação)
    H = np.maximum(0, np.dot(X, W) + b)
    
    # Calcula os pesos de saída (Beta) usando a pseudoinversa (Moore-Penrose)
    H_pinv = np.linalg.pinv(H)
    beta = np.dot(H_pinv, y)
    
    return W, b, beta

def prever_elm(X, W, b, beta):
    """Faz a previsão e arredonda para 0 ou 1"""
    H = np.maximum(0, np.dot(X, W) + b)
    y_pred_continuo = np.dot(H, beta)
    # Limiar em 0.5 para decidir a classe
    return np.where(y_pred_continuo >= 0.5, 1, 0)


# ==========================================
# 2. CARREGAMENTO DOS DADOS
# ==========================================
cancer_data = load_breast_cancer()
X = cancer_data.data
y = cancer_data.target # 0 = Maligno, 1 = Benigno


# ==========================================
# 3. CONFIGURAÇÃO DA VALIDAÇÃO CRUZADA E HIPERPARÂMETROS
# ==========================================
# Vamos testar o SMOTE com 1, 3, 5, 7 e 9 vizinhos
valores_k = [1, 3, 5, 7, 9]

# Configurando o K-Fold para 5 rodadas (80% treino, 20% teste por rodada)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Iniciando busca pelo melhor 'k' no SMOTE...\n")

for k in valores_k:
    acuracias = []
    sensibilidades = [] # Taxa de acerto para classe 0 (Maligno)
    especificidades = [] # Taxa de acerto para classe 1 (Benigno)

    melhor_acuracia_teste = 0
    melhor_W = None
    melhor_W = None
    melhor_b = None
    melhor_beta = None
    melhor_y_teste_pred = None
    melhor_y_teste = None
    melhor_sensibilidade = 0
    melhor_especificidade = 0
    
    for indice_treino, indice_teste in skf.split(X, y):
        # A. Separa os dados da rodada
        X_treino, X_teste = X[indice_treino], X[indice_teste]
        y_treino, y_teste = y[indice_treino], y[indice_teste]
        
        # B. NORMALIZAÇÃO (Essencial para a ELM e para o SMOTE funcionar bem)
        scaler = StandardScaler()
        X_treino_norm = scaler.fit_transform(X_treino)
        # O teste é normalizado com a base do treino (não chama fit_transform aqui!)
        X_teste_norm = scaler.transform(X_teste)
        
        # C. APLICA O SMOTE (APENAS NO TREINO)
        # Estratégia 'auto' iguala o número da classe minoritária à majoritária
        smote = SMOTE(sampling_strategy='auto', k_neighbors=k, random_state=42)
        X_treino_smote, y_treino_smote = smote.fit_resample(X_treino_norm, y_treino)
        

        # --- VERIFICAÇÃO DO TAMANHO DO DATASET ---
        print("\n--- Verificação de Shapes ---")
        print(f"X_treino original: {X_treino_norm.shape} -> {X_treino_norm.shape[0]} pacientes e {X_treino_norm.shape[1]} features")
        print(f"X_treino com SMOTE: {X_treino_smote.shape} -> {X_treino_smote.shape[0]} pacientes e {X_treino_smote.shape[1]} features")
        
        # Conta e exibe a quantidade exata de cada classe no novo dataset
        classes, contagens = np.unique(y_treino_smote, return_counts=True)
        print(f"Distribuição das Classes no novo y: Malignos (0) = {contagens[0]} | Benignos (1) = {contagens[1]}")
        print("-----------------------------\n")
        # D. TREINA A ELM com os dados balanceados e normalizados
        W, b, beta = treinar_elm(X_treino_smote, y_treino_smote, n_oculta=150)
        
        # E. TESTA A ELM apenas nos dados originais de teste
        y_pred = prever_elm(X_teste_norm, W, b, beta)
        
        # F. CALCULA MÉTRICAS
        acuracias.append(accuracy_score(y_teste, y_pred))
        # pos_label=0 avalia o quão bem ele acha os Malignos (Sensibilidade)
        sensibilidades.append(recall_score(y_teste, y_pred, pos_label=0)) 
        # pos_label=1 avalia o quão bem ele acha os Benignos (Especificidade)
        especificidades.append(recall_score(y_teste, y_pred, pos_label=1))

        if acuracias[-1] > melhor_acuracia_teste and melhor_sensibilidade < sensibilidades[-1]:
            melhor_acuracia_teste = acuracias[-1]
            melhor_W = W
            melhor_b = b
            melhor_beta = beta
            melhor_y_teste_pred = y_pred
            melhor_y_teste = y_teste
            melhor_sensibilidade = sensibilidades[-1]
            melhor_especificidade = especificidades[-1]
        
    # Médias da validação cruzada para este 'k'
    print(f"--- Resultados para k={k} vizinhos ---")
    print(f"Acurácia Geral: {np.mean(acuracias)*100:.2f}%")
    print(f"Sensibilidade (Acerto Malignos - Cuidado!): {np.mean(sensibilidades)*100:.2f}%")
    print(f"Especificidade (Acerto Benignos): {np.mean(especificidades)*100:.2f}%\n")

print("\n" + "="*50)
print("Melhor Acurácia de Teste Obtida: {:.2f}%".format(melhor_acuracia_teste * 100))
print(f"Melhor Sensibilidade (Malignos): {melhor_sensibilidade * 100:.2f}%")
print(f"Melhor Especificidade (Benignos): {melhor_especificidade * 100:.2f}%")
print("\nMatriz de Confusão do Melhor Modelo:")
print(confusion_matrix(melhor_y_teste, melhor_y_teste_pred))
print("\n" + "="*50)

