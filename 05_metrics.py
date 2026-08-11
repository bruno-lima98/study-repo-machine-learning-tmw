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

arvore = tree.DecisionTreeClassifier(random_state=42, 
                                     min_samples_leaf=5)

arvore.fit(X=X, y=y)

# %%

arvore_predict = arvore.predict(X)
arvore_predict

df_predict = df_analise[["feliz"]]
df_predict["predict_arvore"] = arvore_predict
df_predict

# %%
# ACURÁCIA
(df_predict["feliz"] == df_predict["predict_arvore"]).mean()

# %%
pd.crosstab(df_predict["feliz"], df_predict["predict_arvore"])