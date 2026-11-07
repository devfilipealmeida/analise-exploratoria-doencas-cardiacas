# 🫀 Heart Disease Classification: Experimentação e MVP

Projeto de estudo de Machine Learning que cobre o ciclo completo de um MVP de modelo: **análise exploratória (EDA)**, **tratamento de dados**, **treinamento de um baseline** e **rastreamento de experimentos com MLflow**, usando o dataset *Heart Disease* da UCI para prever a presença de doença cardíaca.

## 🎯 Objetivo

Prever, a partir de dados clínicos do paciente (idade, pressão, colesterol, resultados de exames etc.), se há **presença de doença cardíaca**, tratando o problema como **classificação binária**.

## 📊 Dataset

- **Fonte:** conjunto consolidado dos quatro datasets da UCI (Cleveland, Hungarian, Switzerland e VA Long Beach), disponível no [Kaggle](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data/data). O original está no [UCI Repository](https://archive.ics.uci.edu/dataset/45/heart+disease).
- **Tamanho:** 920 observações e 16 colunas.
- **Alvo original (`num`):** 0 a 4. Foi binarizado: `0` = sem doença, `1` = com doença (valores > 0).
- **Distribuição das classes:** ~55% com doença e ~45% sem doença (ratio ≈ 0,81, levemente desbalanceado).

| Coluna | Descrição |
|---|---|
| `age` | Idade (anos) |
| `sex` | Sexo |
| `cp` | Tipo de dor no peito |
| `trestbps` | Pressão arterial em repouso (mmHg) |
| `chol` | Colesterol sérico (mg/dl) |
| `fbs` | Glicemia em jejum > 120 mg/dl |
| `restecg` | ECG em repouso |
| `thalch` | Frequência cardíaca máxima alcançada |
| `exang` | Angina induzida por exercício |
| `oldpeak` | Depressão do segmento ST induzida por exercício |
| `slope` | Inclinação do segmento ST de pico |
| `ca` | Nº de vasos principais coloridos por fluoroscopia (0–3) |
| `thal` | Estado talassêmico |
| `num` → `target` | Diagnóstico (alvo) |

> ⚠️ O dataset **não está versionado neste repositório**. Baixe-o no Kaggle e coloque em `data/heart_disease_uci.csv`.

## 🔬 Etapas do projeto

### 1. Análise Exploratória (EDA)
- Informações gerais e estatísticas descritivas
- Análise de **valores ausentes** (algumas colunas têm mais de 50% de missing, como `ca` e `thal`)
- Distribuição e balanceamento da variável alvo
- Detecção de **outliers** com boxplots e z-score
- Verificação de anomalias e valores inválidos no contexto clínico
- Distribuições, assimetria (skewness) e correlações

### 2. Preparação dos dados
- **Imputação de missing values**
  - Numéricas (`trestbps`, `chol`, `thalch`, `oldpeak`, `ca`): **mediana**
  - Categóricas (`thal`, `slope`, `fbs`, `exang`, `restecg`): **moda**
  - As estatísticas são calculadas **somente no conjunto de treino**, para evitar *data leakage*.
- **One-hot encoding** (`pd.get_dummies(drop_first=True)`) em `sex`, `cp`, `restecg`, `slope` e `thal`
- Remoção das colunas `id` e `dataset`
- Divisão treino/teste 80/20 com `random_state=42`

### 3. Modelagem (baseline)
- **Regressão Logística** (`sklearn.linear_model.LogisticRegression`)
- Métricas avaliadas: acurácia (treino e teste), F1, precisão, recall e diferença treino − teste como indicador de *overfitting*

### 4. Rastreamento de experimentos com MLflow
- Experimento: `heart_disease_classification`
- Run: `logistic_regression_baseline`
- Registro de métricas e do modelo treinado

### 5. Persistência
- DataFrame pré-processado salvo em CSV
- Modelo salvo via MLflow e `joblib`

## 📈 Resultados

| Modelo | Acurácia (treino) | Acurácia (teste) | F1 | Precisão | Recall |
|---|---|---|---|---|---|
| Regressão Logística (baseline) | _preencher_ | _preencher_ | _preencher_ | _preencher_ | _preencher_ |

## 🧠 Principais aprendizados

- **Data leakage:** calcular mediana/moda no dataset inteiro "vaza" informação do teste para o treino. O correto é aprender essas estatísticas só no treino.
- **Outlier ≠ erro:** valores impossíveis (ex.: `chol = 0`, `trestbps = 0`) são dados faltantes disfarçados; valores extremos plausíveis podem ser mantidos; categorias raras (ex.: `ca = 3`) não são outliers de verdade.
- **Z-score vs. IQR:** o z-score é sensível a valores extremos que inflam o desvio padrão. O IQR (usado no boxplot) é mais robusto em dados com cauda longa.
- **`fillna(inplace=True)` em coluna selecionada não funciona no pandas com Copy-on-Write.** O correto é `df[col] = df[col].fillna(valor)`.
- **Modelos lineares não aceitam `NaN`:** é preciso imputar ou usar modelos que tratem ausentes nativamente (ex.: `HistGradientBoostingClassifier`).

## 🚀 Como executar

```bash
# 1. Clone o repositório
git clone https://github.com/devfilipealmeida/<nome-do-repositorio>.git
cd <nome-do-repositorio>

# 2. (Opcional) Crie um ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows

# 3. Instale as dependências
pip install pandas numpy matplotlib seaborn scipy scikit-learn mlflow joblib jupyter

# 4. Coloque o dataset em data/heart_disease_uci.csv

# 5. Abra o notebook
jupyter notebook aula_2.ipynb
```

Para visualizar os experimentos no MLflow:

```bash
mlflow ui
# acesse http://localhost:5000
```

## 🗂️ Estrutura sugerida

```
.
├── data/
│   ├── heart_disease_uci.csv                # dataset original (não versionado)
│   └── heart_disease_uci_preprocessed.csv   # dataset pré-processado
├── models/
│   └── baseline_model.joblib                # modelo treinado
├── analise_exploratoria_doencas_cardiacas.ipynb  # notebook principal
└── README.md
```

## 🛠️ Tecnologias

Python 3.13 · pandas · NumPy · Matplotlib · Seaborn · SciPy · scikit-learn · MLflow · joblib · Jupyter

## 🔭 Próximos passos

- [ ] Tratar `chol = 0` e `trestbps = 0` como valores ausentes antes da imputação
- [ ] Usar `Pipeline` + `ColumnTransformer` (imputação, escala e encoding dentro do pipeline)
- [ ] Padronizar as features com `StandardScaler`
- [ ] Comparar outros algoritmos (Random Forest, Gradient Boosting, SVM)
- [ ] Busca de hiperparâmetros e validação cruzada estratificada
- [ ] Analisar matriz de confusão e curva ROC
- [ ] Interpretabilidade das features (coeficientes, SHAP)

## 📚 Referências

- [UCI Machine Learning Repository: Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)
- [Dataset no Kaggle](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data/data)
- [Documentação do scikit-learn: Imputação](https://scikit-learn.org/stable/modules/impute.html)
- [Documentação do MLflow](https://mlflow.org/docs/latest/index.html)

## 👤 Autor

**Filipe Almeida**: [github.com/devfilipealmeida](https://github.com/devfilipealmeida)
