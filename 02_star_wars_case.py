# %%
import pandas as pd

# %%
df = pd.read_parquet(
    "data/dados_clones.parquet",
    engine="fastparquet"
)

df.head()

# %%
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# %%
features = [
    "massa(em_kilos)", "general_jedi_encarregado",
    "estatura(cm)", "distância_ombro_a_ombro", "tamanho_do_crânio",
    "tamanho_dos_pés", "tempo_de_existência(em_meses)",
    ]

target = "status"

# %%
X = df[features]

y = df[target]

# %%
X.head(2)

# %%
y.head(2)

# %%
for i in X.columns:
    if X[i].dtype == "O":
        print(i, X[i].unique())

# %%
X = X.replace({
    "Yoda": 1, 
    "Shaak Ti": 2,
    "Obi-Wan Kenobi": 3, 
    "Aayla Secura": 4, 
    "Mace Windu": 5,
    "Tipo 1": 1, 
    "Tipo 2": 2, 
    "Tipo 3": 3, 
    "Tipo 4": 4,
    "Tipo 5": 5
})

X.head()

# %%
from sklearn import tree

# %%
model = tree.DecisionTreeClassifier()

# %%
model.fit(X=X, y=y)

# %%
import matplotlib.pyplot as plt

plt.figure(dpi=400)

tree.plot_tree(
    model,
    feature_names = features,
    class_names = model.classes_,
    filled = True,
    max_depth=3,
)

plt.title("Árvore com General")
plt.show()

# %%
features_2 = [
    "massa(em_kilos)", "estatura(cm)", "distância_ombro_a_ombro", 
    "tamanho_do_crânio", "tamanho_dos_pés", "tempo_de_existência(em_meses)"
    ]

X2 = df[features_2]

X2 = X2.replace({
    "Tipo 1": 1, 
    "Tipo 2": 2, 
    "Tipo 3": 3, 
    "Tipo 4": 4,
    "Tipo 5": 5
})

# %%
model_2 = tree.DecisionTreeClassifier()

model_2.fit(X=X2, y=y)

# %%
plt.figure(dpi=400)

tree.plot_tree(
    model,
    feature_names = features,
    class_names = model.classes_,
    filled = True,
    max_depth=3,
)

plt.title("Árvore sem General")
plt.show()