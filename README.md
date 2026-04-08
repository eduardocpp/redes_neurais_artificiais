# Redes Neurais Artificiais — Fundamentos e Implementações

Repositório com implementações práticas de modelos clássicos de redes neurais artificiais, desenvolvidas em Python com foco didático. Os experimentos cobrem desde o Perceptron simples até Extreme Learning Machines (ELM), passando por aproximação polinomial, superfícies de erro e técnicas de balanceamento de dados.

---

## Introdução

Redes neurais artificiais são modelos computacionais inspirados no funcionamento biológico dos neurônios. A ideia central é simples: assim como neurônios biológicos recebem sinais, os processam e transmitem uma resposta, um neurônio artificial recebe entradas numéricas, aplica pesos e uma função de ativação, e produz uma saída. Quando organizados em camadas — entrada, oculta(s) e saída — esses neurônios formam redes capazes de aprender padrões complexos diretamente a partir dos dados.

O aprendizado ocorre pelo ajuste iterativo dos pesos sinápticos, buscando minimizar o erro entre a saída produzida pela rede e a saída desejada. Esse processo é o que permite a uma rede neural generalizar para dados que nunca viu durante o treinamento.

## Base Matemática

### O Neurônio Artificial

O modelo básico de um neurônio artificial calcula uma soma ponderada das entradas e aplica uma função de ativação:

```
u = Σ (wᵢ · xᵢ) + b
y = φ(u)
```

Onde **x** é o vetor de entradas, **w** é o vetor de pesos, **b** é o bias e **φ** é a função de ativação (degrau, sigmoide, tanh, ReLU, etc.).

### Regra de Aprendizado do Perceptron

O Perceptron ajusta seus pesos com base no erro entre a saída prevista e a desejada:

```
w(t+1) = w(t) + η · (y_desejado - y_previsto) · x
```

Onde **η** é a taxa de aprendizado. Esse processo converge para dados linearmente separáveis.

### Pseudoinversa de Moore-Penrose

Para modelos como a ELM e a aproximação polinomial, os pesos da camada de saída são calculados analiticamente sem iteração:

```
β = H⁺ · Y
```

Onde **H** é a matriz de ativações da camada oculta e **H⁺** é sua pseudoinversa. Isso resolve o sistema no sentido de mínimos quadrados em uma única operação.

### Aproximação Polinomial e Overfitting

A aproximação polinomial ilustra o compromisso entre complexidade do modelo e capacidade de generalização. O risco real de um modelo é limitado superiormente por:

```
R ≤ R_empírico + √(h / N)
```

Onde **h** mede a complexidade do modelo e **N** o tamanho do dataset. Modelos muito complexos para poucos dados levam ao overfitting; modelos simples demais levam ao underfitting.

---

## Estrutura do Repositório

### Perceptron e Portas Lógicas

**`ORANDXOR.py`** — Implementação do Perceptron para aprender as portas lógicas OR, AND e XOR. Demonstra a limitação clássica do Perceptron simples com o XOR (problema não linearmente separável) e resolve o problema usando composição de saídas das portas OR e AND como entrada para um segundo estágio, formando uma arquitetura multicamada rudimentar. Inclui também uma implementação do ADALINE para comparação.

### Projeção Não-Linear e Separabilidade

**`exercicio2_questao1.py`** — Demonstra como uma transformação não-linear pode tornar um problema de classificação circular (não linearmente separável no espaço original) em um problema linearmente separável em um espaço de dimensão superior. A projeção utilizada é `Z = x² + y²`, que transforma a fronteira circular em um limiar plano no espaço projetado. Inclui visualizações 2D e 3D.

### Superfície de Erro

**`superficie_erro.py`** — Visualização 3D da superfície de erro quadrático para uma regressão linear simples (`y = w1·x + w0`), varrendo os valores de `w0` e `w1` em uma grade. Permite observar o formato da superfície (convexa, com um único mínimo global) e entender intuitivamente o que o processo de otimização busca.

**`es.py`** — Plotagem da superfície `z = 0.5x² + 3x - y` para visualização de funções em 3D.

### Aproximação Polinomial

**`polinomios.py`** — Estudo prático de overfitting e underfitting usando aproximação polinomial com pseudoinversa. Compara o comportamento de polinômios de grau 2 e grau 4 ajustados a datasets de diferentes tamanhos (50 e 500 amostras), demonstrando como o aumento do número de amostras mitiga o overfitting em modelos mais complexos.

**`exercicio2_questao3.py`** — Extensão do estudo anterior: varia o grau do polinômio de 1 a 8 para 10 e 100 amostras, gerando gráficos comparativos com a função geradora original. Permite identificar visualmente em quais graus ocorre underfitting e overfitting.

**`Polinomios_DL.py`** — Experimento com treinamento de um perceptron com ativação ReLU para aprender a mapear entradas para os coeficientes de uma função quadrática, explorando a decomposição `[x², x, 1]` como camada oculta.

### Rede Neural Multicamada (Exemplo Didático)

**`exemplo1.py`** — Exemplo mínimo de uma rede neural feedforward de duas camadas (oculta + saída) com ativação `tanh`. Demonstra o fluxo de dados: entrada → pesos → ativação → saída → cálculo do erro. Serve como base conceitual para entender a propagação direta (forward pass).

### Classificação com Perceptron

**`Exercicio4/Q1.py`** — Classificação binária de duas distribuições gaussianas 2D com o Perceptron. Gera dados com médias `[2,2]` e `[4,4]` e variância `0.16`. Inclui visualização da fronteira de decisão linear no espaço 2D e a superfície de decisão em degrau no espaço 3D.

**`Exercicio4/Q2.py`** — Avaliação do Perceptron com divisão treino/teste (70/30). Apresenta acurácia e matriz de confusão para ambos os conjuntos, usando os mesmos dados gaussianos do Q1.

**`Exercicio4/Q3.py`** — Aplicação do Perceptron na base Iris (classificação binária: Setosa vs. não-Setosa). Executa 100 rodadas de treinamento com divisão estratificada e reporta acurácia média e variância nos conjuntos de treino e teste.

**`Exercicio4/Q4.py`** — Aplicação do Perceptron na base Breast Cancer (classificação maligno/benigno). Demonstra as limitações do Perceptron simples em dados de alta dimensionalidade (30 features) que não são perfeitamente separáveis linearmente.

### Extreme Learning Machine (ELM)

**`ELM.py`** — Implementação completa de uma ELM para classificação binária de dados gaussianos 2D. A camada oculta possui pesos aleatórios fixos com ativação `tanh`, e os pesos de saída são calculados via pseudoinversa. Inclui visualização da fronteira de decisão 2D (contorno), avaliação em dados de teste e plotagem da superfície de decisão em 3D.

**`ELM_breast_cancer/ELM_Breast_Cancer.py`** — ELM aplicada à base Breast Cancer com 1000 iterações, variando o número de neurônios ocultos de 1 a 1000. Registra o melhor modelo encontrado e reporta acurácia média, variância e matriz de confusão.

**`ELM_breast_cancer/ELM_breast_cancer_normal.py`** — Versão aprimorada do experimento anterior com normalização dos dados (StandardScaler) aplicada corretamente dentro do loop de treino/teste, evitando data leakage. Adiciona bias após a normalização e salva o gabarito correto para a matriz de confusão do melhor modelo.

**`ELM_breast_cancer/ELM_breast_cancer_SMOTE.py`** — ELM com balanceamento de classes via SMOTE (Synthetic Minority Over-sampling Technique). Utiliza validação cruzada estratificada (5-fold), normalização, ativação ReLU na camada oculta e busca pelo melhor valor de `k` vizinhos no SMOTE (1, 3, 5, 7 e 9). Reporta acurácia, sensibilidade (acerto em malignos) e especificidade (acerto em benignos).

---

## Dependências

```
numpy
matplotlib
scikit-learn
imbalanced-learn
```

Instalação:

```bash
pip install numpy matplotlib scikit-learn imbalanced-learn
```

---

## Como Executar

Cada script é independente e pode ser executado diretamente:

```bash
python nome_do_script.py
```

Os scripts que geram gráficos abrirão janelas do matplotlib. Os demais imprimem resultados no terminal.
