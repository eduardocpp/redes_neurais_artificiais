import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

def tanh(x):
    return np.tanh(x)

def elm_train(X, y, n_hidden):
    n_features = X.shape[1]
    W = np.random.randn(n_features, n_hidden)
    b = np.random.randn(1, n_hidden)
    H = tanh(X @ W + b)
    H_pinv = np.linalg.pinv(H)
    beta = H_pinv @ y
    return W, b, beta

def elm_predict(X, W, b, beta):
    H = tanh(X @ W + b)
    y_pred = H @ beta
    return np.sign(y_pred)

def accuracy(y_true, y_pred):
    return np.mean(y_true.flatten() == y_pred.flatten())

def generate_xor(n_per_cluster=100, std=0.1):
    centers = np.array([[0,0],[0,1],[1,0],[1,1]])
    labels = np.array([-1, 1, 1, -1])
    X, y = [], []
    for c, l in zip(centers, labels):
        X.append(c + np.random.randn(n_per_cluster, 2) * std)
        y.append(np.full(n_per_cluster, l))
    return np.vstack(X), np.concatenate(y).reshape(-1,1)

########################################
# EXERCICIO 1 - ELM no problema XOR
########################################

neurons_list = [2, 5, 10, 50, 100, 500, 1000]
stds = [0.1, 0.2, 0.3]
k_folds = 10

fig, axes = plt.subplots(1, len(stds), figsize=(6*len(stds), 5))

for idx, std in enumerate(stds):
    X, y = generate_xor(n_per_cluster=200, std=std)
    train_errors = []
    test_errors = []

    for n_h in neurons_list:
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        fold_train_err = []
        fold_test_err = []
        for train_idx, test_idx in kf.split(X):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            W, b, beta = elm_train(X_tr, y_tr, n_h)
            pred_tr = elm_predict(X_tr, W, b, beta)
            pred_te = elm_predict(X_te, W, b, beta)
            fold_train_err.append(1 - accuracy(y_tr, pred_tr))
            fold_test_err.append(1 - accuracy(y_te, pred_te))
        train_errors.append(np.mean(fold_train_err))
        test_errors.append(np.mean(fold_test_err))

    ax = axes[idx]
    ax.plot(neurons_list, train_errors, 'o-', label='Erro Treino')
    ax.plot(neurons_list, test_errors, 's-', label='Erro Teste')
    ax.set_xscale('log')
    ax.set_xlabel('Número de Neurônios')
    ax.set_ylabel('Erro Médio')
    ax.set_title(f'XOR - std={std}')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig('Exercicio5/exercicio1_xor.png', dpi=150)
plt.close()

print("=== EXERCICIO 1 - XOR ===")
for std in stds:
    X, y = generate_xor(n_per_cluster=200, std=std)
    print(f"\nstd = {std}")
    for n_h in neurons_list:
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        tr_e, te_e = [], []
        for train_idx, test_idx in kf.split(X):
            W, b, beta = elm_train(X[train_idx], y[train_idx], n_h)
            tr_e.append(1 - accuracy(y[train_idx], elm_predict(X[train_idx], W, b, beta)))
            te_e.append(1 - accuracy(y[test_idx], elm_predict(X[test_idx], W, b, beta)))
        print(f"  Neuronios={n_h:5d} | Erro Treino={np.mean(tr_e):.4f} | Erro Teste={np.mean(te_e):.4f}")

########################################
# EXERCICIO 2 - ELMs em bases reais
########################################

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X_bc = data.data
y_bc = (data.target * 2 - 1).reshape(-1, 1)

scaler_bc = MinMaxScaler()
X_bc = scaler_bc.fit_transform(X_bc)

neurons_real = [5, 10, 30, 50, 100, 300]
n_runs = 10

print("\n=== EXERCICIO 2 - Breast Cancer ===")
bc_train_accs = {n: [] for n in neurons_real}
bc_test_accs = {n: [] for n in neurons_real}

for run in range(n_runs):
    perm = np.random.permutation(len(X_bc))
    split = int(0.7 * len(X_bc))
    X_tr, X_te = X_bc[perm[:split]], X_bc[perm[split:]]
    y_tr, y_te = y_bc[perm[:split]], y_bc[perm[split:]]
    for n_h in neurons_real:
        W, b, beta = elm_train(X_tr, y_tr, n_h)
        bc_train_accs[n_h].append(accuracy(y_tr, elm_predict(X_tr, W, b, beta)))
        bc_test_accs[n_h].append(accuracy(y_te, elm_predict(X_te, W, b, beta)))

for n_h in neurons_real:
    tr_m, tr_s = np.mean(bc_train_accs[n_h]), np.std(bc_train_accs[n_h])
    te_m, te_s = np.mean(bc_test_accs[n_h]), np.std(bc_test_accs[n_h])
    print(f"  Neuronios={n_h:4d} | Treino={tr_m:.4f}+-{tr_s:.4f} | Teste={te_m:.4f}+-{te_s:.4f}")

# Perceptron para Breast Cancer
def perceptron_train(X, y, lr=0.01, max_iter=1000):
    n = X.shape[1]
    w = np.zeros(n)
    b_p = 0.0
    for _ in range(max_iter):
        for i in range(len(X)):
            pred = np.sign(X[i] @ w + b_p)
            if pred == 0:
                pred = -1
            if pred != y[i, 0]:
                w += lr * y[i, 0] * X[i]
                b_p += lr * y[i, 0]
    return w, b_p

def perceptron_predict(X, w, b_p):
    pred = np.sign(X @ w + b_p)
    pred[pred == 0] = -1
    return pred.reshape(-1, 1)

print("\n--- Perceptron - Breast Cancer ---")
p_train_accs_bc, p_test_accs_bc = [], []
for run in range(n_runs):
    perm = np.random.permutation(len(X_bc))
    split = int(0.7 * len(X_bc))
    X_tr, X_te = X_bc[perm[:split]], X_bc[perm[split:]]
    y_tr, y_te = y_bc[perm[:split]], y_bc[perm[split:]]
    w, b_p = perceptron_train(X_tr, y_tr)
    p_train_accs_bc.append(accuracy(y_tr, perceptron_predict(X_tr, w, b_p)))
    p_test_accs_bc.append(accuracy(y_te, perceptron_predict(X_te, w, b_p)))
print(f"  Perceptron | Treino={np.mean(p_train_accs_bc):.4f}+-{np.std(p_train_accs_bc):.4f} | Teste={np.mean(p_test_accs_bc):.4f}+-{np.std(p_test_accs_bc):.4f}")

# Statlog Heart
import urllib.request
import os

heart_path = 'Exercicio5/heart.dat'
if not os.path.exists(heart_path):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat"
    try:
        urllib.request.urlretrieve(url, heart_path)
    except:
        pass

if os.path.exists(heart_path):
    heart_data = np.loadtxt(heart_path)
    X_heart = heart_data[:, :-1]
    y_heart_raw = heart_data[:, -1]
    y_heart = np.where(y_heart_raw == 1, -1, 1).reshape(-1, 1)
    scaler_h = MinMaxScaler()
    X_heart = scaler_h.fit_transform(X_heart)

    print("\n=== EXERCICIO 2 - Statlog Heart ===")
    h_train_accs = {n: [] for n in neurons_real}
    h_test_accs = {n: [] for n in neurons_real}

    for run in range(n_runs):
        perm = np.random.permutation(len(X_heart))
        split = int(0.7 * len(X_heart))
        X_tr, X_te = X_heart[perm[:split]], X_heart[perm[split:]]
        y_tr, y_te = y_heart[perm[:split]], y_heart[perm[split:]]
        for n_h in neurons_real:
            W, b, beta = elm_train(X_tr, y_tr, n_h)
            h_train_accs[n_h].append(accuracy(y_tr, elm_predict(X_tr, W, b, beta)))
            h_test_accs[n_h].append(accuracy(y_te, elm_predict(X_te, W, b, beta)))

    for n_h in neurons_real:
        tr_m, tr_s = np.mean(h_train_accs[n_h]), np.std(h_train_accs[n_h])
        te_m, te_s = np.mean(h_test_accs[n_h]), np.std(h_test_accs[n_h])
        print(f"  Neuronios={n_h:4d} | Treino={tr_m:.4f}+-{tr_s:.4f} | Teste={te_m:.4f}+-{te_s:.4f}")

    print("\n--- Perceptron - Statlog Heart ---")
    p_train_accs_h, p_test_accs_h = [], []
    for run in range(n_runs):
        perm = np.random.permutation(len(X_heart))
        split = int(0.7 * len(X_heart))
        X_tr, X_te = X_heart[perm[:split]], X_heart[perm[split:]]
        y_tr, y_te = y_heart[perm[:split]], y_heart[perm[split:]]
        w, b_p = perceptron_train(X_tr, y_tr)
        p_train_accs_h.append(accuracy(y_tr, perceptron_predict(X_tr, w, b_p)))
        p_test_accs_h.append(accuracy(y_te, perceptron_predict(X_te, w, b_p)))
    print(f"  Perceptron | Treino={np.mean(p_train_accs_h):.4f}+-{np.std(p_train_accs_h):.4f} | Teste={np.mean(p_test_accs_h):.4f}+-{np.std(p_test_accs_h):.4f}")
else:
    print("\nNao foi possivel baixar a base Statlog Heart. Baixe manualmente o arquivo heart.dat e coloque no mesmo diretorio.")

# Graficos Exercicio 2
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

tr_means = [np.mean(bc_train_accs[n]) for n in neurons_real]
tr_stds = [np.std(bc_train_accs[n]) for n in neurons_real]
te_means = [np.mean(bc_test_accs[n]) for n in neurons_real]
te_stds = [np.std(bc_test_accs[n]) for n in neurons_real]

axes2[0].errorbar(neurons_real, tr_means, yerr=tr_stds, fmt='o-', label='ELM Treino', capsize=3)
axes2[0].errorbar(neurons_real, te_means, yerr=te_stds, fmt='s-', label='ELM Teste', capsize=3)
axes2[0].axhline(y=np.mean(p_train_accs_bc), color='green', linestyle='--', label=f'Perceptron Treino ({np.mean(p_train_accs_bc):.4f})')
axes2[0].axhline(y=np.mean(p_test_accs_bc), color='red', linestyle='--', label=f'Perceptron Teste ({np.mean(p_test_accs_bc):.4f})')
axes2[0].set_xlabel('Número de Neurônios')
axes2[0].set_ylabel('Acurácia')
axes2[0].set_title('Breast Cancer - ELM vs Perceptron')
axes2[0].legend(fontsize=8)
axes2[0].grid(True)

if os.path.exists(heart_path):
    tr_means_h = [np.mean(h_train_accs[n]) for n in neurons_real]
    tr_stds_h = [np.std(h_train_accs[n]) for n in neurons_real]
    te_means_h = [np.mean(h_test_accs[n]) for n in neurons_real]
    te_stds_h = [np.std(h_test_accs[n]) for n in neurons_real]

    axes2[1].errorbar(neurons_real, tr_means_h, yerr=tr_stds_h, fmt='o-', label='ELM Treino', capsize=3)
    axes2[1].errorbar(neurons_real, te_means_h, yerr=te_stds_h, fmt='s-', label='ELM Teste', capsize=3)
    axes2[1].axhline(y=np.mean(p_train_accs_h), color='green', linestyle='--', label=f'Perceptron Treino ({np.mean(p_train_accs_h):.4f})')
    axes2[1].axhline(y=np.mean(p_test_accs_h), color='red', linestyle='--', label=f'Perceptron Teste ({np.mean(p_test_accs_h):.4f})')
    axes2[1].set_xlabel('Número de Neurônios')
    axes2[1].set_ylabel('Acurácia')
    axes2[1].set_title('Statlog Heart - ELM vs Perceptron')
    axes2[1].legend(fontsize=8)
    axes2[1].grid(True)

plt.tight_layout()
plt.savefig('Exercicio5/exercicio2_real.png', dpi=150)
plt.close()

print("\nGraficos salvos em Exercicio5/exercicio1_xor.png e Exercicio5/exercicio2_real.png")