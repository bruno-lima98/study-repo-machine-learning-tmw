# %%
import pandas as pd

# %%
df = pd.read_excel("data/dados_cerveja_nota.xlsx")
df.head()

# %%
# REGRESSÃO LINEAR

from sklearn import linear_model

# %%
X = df[["cerveja"]] # isso é sempre uma matriz (dataframe) 
y = df["nota"]      # isso é sempre um vetor (series)

# %%
modelo_regressao = linear_model.LinearRegression()

modelo_regressao.fit(X=X, y=y)

# %%
a, b = round(modelo_regressao.intercept_,4), round(modelo_regressao.coef_[0],4)
print("Coeficiente A:", a)
print("Coeficiente B:", b)

# y = a + b.x

# %%
predict = modelo_regressao.predict(X.drop_duplicates())
predict

# %%
import matplotlib.pyplot as plt

plt.plot(X["cerveja"], y, "o")
plt.grid(True)
plt.title("Relação Cerveja x Nota")
plt.xlabel("Cerveja")
plt.ylabel("Nota")

plt.plot(X.drop_duplicates()["cerveja"], predict) # RETA COM OS PREDICTS

plt.legend(["Observado", f"y = {a} + {b}x"])

plt.show()

# %%
from sklearn import tree

# %%
modelo_arvore = tree.DecisionTreeRegressor(random_state=42)
modelo_arvore.fit(X=X, y=y)
predict_arvore = modelo_arvore.predict(X.drop_duplicates())

modelo_arvore_d2 = tree.DecisionTreeRegressor(random_state=42, max_depth=2)
modelo_arvore_d2.fit(X=X, y=y)
predict_arvore_d2 = modelo_arvore_d2.predict(X.drop_duplicates())

# %%

plt.plot(X["cerveja"], y, "o")
plt.grid(True)
plt.title("Relação Cerveja x Nota")
plt.xlabel("Cerveja")
plt.ylabel("Nota")

plt.plot(X.drop_duplicates()["cerveja"], predict) # RETA COM OS PREDICTS
plt.plot(X.drop_duplicates()["cerveja"], predict_arvore)
plt.plot(X.drop_duplicates()["cerveja"], predict_arvore_d2)

plt.legend([
    "Observado", 
    f"y = {a} + {b}x",
    "Árvore Full",
    "Árvore Depth = 2"
    ])

# %%
tree.plot_tree(
    modelo_arvore_d2,
    feature_names=["cerveja"],    
    filled=True)

plt.show()