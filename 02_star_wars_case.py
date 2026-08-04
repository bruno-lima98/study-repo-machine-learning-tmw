# %%
import pandas as pd

# %%
df = pd.read_parquet(
    "data/dados_clones.parquet",
    engine="fastparquet"
)

df.head()

# %%
