# %%
import pandas as pd


# %%
df = pd.read_csv("data/dados_comunidade.csv")
df.head()

# %%
df = df.replace({"Sim": 1, "Não": 0})
df.head()

# %%

dummy_vars = [
    "Como conheceu o Téo Me Why?",
    "Quantos cursos acompanhou do Téo Me Why?",
    "Estado que mora atualmente",
    "Área de Formação",
    "Tempo que atua na área de dados",
    "Posição da cadeira (senioridade)"
]

df_analise = pd.get_dummies(df[dummy_vars]).astype(int)

# %%
num_vars = [
    "Curte games?",
    "Curte futebol?",
    "Curte livros?",
    "Curte jogos de tabuleiro?",
    "Curte jogos de fórmula 1?",
    "Curte jogos de MMA?",
    "Idade"
]

df_analise[num_vars] = df[num_vars].copy()

df_analise.head()

# %%
df_analise["feliz"] = df["Você se considera uma pessoa feliz?"]
df_analise.head()

# %%
features = df_analise.columns[:-1].tolist()

X = df_analise[features]
y = df_analise["feliz"]

# %%

from sklearn import tree
from sklearn import naive_bayes
from sklearn import linear_model

arvore = tree.DecisionTreeClassifier(random_state=42, 
                                     min_samples_leaf=5)

arvore.fit(X=X, y=y)

naive = naive_bayes.GaussianNB()
naive.fit(X=X, y=y)

lin_reg = linear_model.LogisticRegression()
lin_reg.fit(X=X, y=y)

# %%

df_predict = df_analise[["feliz"]].copy()

# ÁRVORE DE DECISÃO
arvore_predict = arvore.predict(X)

df_predict["predict_arvore"] = arvore_predict
df_predict["proba_arvore"] = arvore.predict_proba(X)[:,1]     

# NAIVE BAYES
naive_predict = naive.predict(X)

df_predict["predict_naive"] = naive_predict
df_predict["proba_naive"] = naive.predict_proba(X)[:,1] 

# REGRESSÃO LOGÍSTICA
lin_reg_predict = lin_reg.predict(X)

df_predict["predict_lin_reg"] = lin_reg_predict
df_predict["proba_lin_reg"] = lin_reg.predict_proba(X)[:,1] 


# %%
# ACURÁCIA
(df_predict["feliz"] == df_predict["predict_arvore"]).mean()

# %%
pd.crosstab(df_predict["feliz"], df_predict["predict_arvore"])
# %%

from sklearn import metrics

acc_arvore = metrics.accuracy_score(df_predict["feliz"], df_predict["predict_arvore"])
precision_arvore = metrics.precision_score(df_predict["feliz"], df_predict["predict_arvore"])
recall_arvore = metrics.recall_score(df_predict["feliz"], df_predict["predict_arvore"])
roc_arvore = metrics.roc_curve(df_predict["feliz"], df_predict["proba_arvore"])
auc_arvore = metrics.roc_auc_score(df_predict["feliz"], df_predict["proba_arvore"])

acc_naive = metrics.accuracy_score(df_predict["feliz"], df_predict["predict_naive"])
precision_naive = metrics.precision_score(df_predict["feliz"], df_predict["predict_naive"])
recall_naive = metrics.recall_score(df_predict["feliz"], df_predict["predict_naive"])
roc_naive = metrics.roc_curve(df_predict["feliz"], df_predict["proba_naive"])
auc_naive = metrics.roc_auc_score(df_predict["feliz"], df_predict["proba_naive"])

acc_lin_reg = metrics.accuracy_score(df_predict["feliz"], df_predict["predict_lin_reg"])
precision_lin_reg = metrics.precision_score(df_predict["feliz"], df_predict["predict_lin_reg"])
recall_lin_reg = metrics.recall_score(df_predict["feliz"], df_predict["predict_lin_reg"])
roc_lin_reg = metrics.roc_curve(df_predict["feliz"], df_predict["proba_lin_reg"])
auc_lin_reg = metrics.roc_auc_score(df_predict["feliz"], df_predict["proba_lin_reg"])

# %%
import matplotlib.pyplot as plt

# %%
plt.plot(roc_arvore[0], roc_arvore[1], "o-")
plt.plot(roc_naive[0], roc_naive[1], "o-")
plt.plot(roc_lin_reg[0], roc_lin_reg[1], "o-")


plt.grid(True)
plt.title("ROC Curve")
plt.xlabel("1 - Especificidade")
plt.ylabel("Recall")

plt.legend([f"Árvore: {auc_arvore:.2f}", f"Naive Bayes: {auc_naive:.2f}", f"Regressão Logística: {auc_lin_reg:.2f}"])

plt.show()

# %%