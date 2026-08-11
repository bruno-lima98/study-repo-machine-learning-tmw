# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
df = pd.read_excel("data/dados_cerveja_nota.xlsx")
df.head()

# %%
df["aprovado"] = (df["nota"] > 5).astype(int)
df

# %%

plt.plot(df["cerveja"], df["aprovado"], "o", color="royalblue")
plt.grid(True)
plt.xlabel("Cerveja")
plt.ylabel("Aprovado")
plt.title("Aprovação das Cervejas")

plt.show()

# %% 
from sklearn import linear_model

reg = linear_model.LogisticRegression(penalty=None, fit_intercept=True)

reg.fit(df[["cerveja"]], df["aprovado"])

# %%

reg_predict = reg.predict(df[["cerveja"]].drop_duplicates())
reg_prob = reg.predict_proba(df[["cerveja"]].drop_duplicates())[:,1]

reg_predict

# %%
plt.plot(df["cerveja"], df["aprovado"], "o", color="royalblue")
plt.grid(True)
plt.xlabel("Cerveja")
plt.ylabel("Aprovado")
plt.title("Aprovação das Cervejas")
plt.hlines(0.5, xmin=1, xmax=9, linestyles="--", colors="black")

plt.plot(df[["cerveja"]].drop_duplicates(), reg_predict, color="red")
plt.plot(df[["cerveja"]].drop_duplicates(), reg_prob, color="green")

plt.legend(["Observação", "Baseline (50%)", "Reg Predict", "Reg Proba"])
plt.show()

# %%
from sklearn import tree

arvore_full = tree.DecisionTreeClassifier(random_state=42, max_depth=2)
arvore_full.fit(df[["cerveja"]], df["aprovado"])
arvore_full_predict = arvore_full.predict(df[["cerveja"]].drop_duplicates())
arvore_full_proba = arvore_full.predict_proba(df[["cerveja"]].drop_duplicates())[:,1]

# %%
plt.plot(df["cerveja"], df["aprovado"], "o", color="royalblue")
plt.grid(True)
plt.xlabel("Cerveja")
plt.ylabel("Aprovado")
plt.title("Aprovação das Cervejas")
plt.hlines(0.5, xmin=1, xmax=9, linestyles="--", colors="black")

plt.plot(df[["cerveja"]].drop_duplicates(), reg_predict, color="red")
plt.plot(df[["cerveja"]].drop_duplicates(), reg_prob, color="green")

plt.plot(df[["cerveja"]].drop_duplicates(), arvore_full_predict, color="orange")
plt.plot(df[["cerveja"]].drop_duplicates(), arvore_full_proba, color="magenta")

plt.legend([
    "Observação", 
    "Baseline (50%)", 
    "Reg Predict", 
    "Reg Proba",
    "Árvore Full",
    "Árvore Full Proba"
    ])
plt.show()

# %%
from sklearn import naive_bayes

nb = naive_bayes.GaussianNB()
nb.fit(df[["cerveja"]], df["aprovado"])
naive_predict = nb.predict(df[["cerveja"]].drop_duplicates())
naive_proba = nb.predict_proba(df[["cerveja"]].drop_duplicates())[:,1]
# %%
plt.plot(df["cerveja"], df["aprovado"], "o", color="royalblue")
plt.grid(True)
plt.xlabel("Cerveja")
plt.ylabel("Aprovado")
plt.title("Aprovação das Cervejas")
plt.hlines(0.5, xmin=1, xmax=9, linestyles="--", colors="black")

plt.plot(df[["cerveja"]].drop_duplicates(), reg_predict, color="red")
plt.plot(df[["cerveja"]].drop_duplicates(), reg_prob, color="green")

plt.plot(df[["cerveja"]].drop_duplicates(), naive_predict, color="orange")
plt.plot(df[["cerveja"]].drop_duplicates(), naive_proba, color="magenta")

plt.legend([
    "Observação", 
    "Baseline (50%)", 
    "Reg Predict", 
    "Reg Proba",
    "Nayve Bayes",
    "Nayve Bayes Proba"
    ])
plt.show()