"""Analysis of CAMME data"""

# %%
import logging

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level="INFO",
)

import time
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analysis.process_camme import preprocess
from analysis.retrieve_data import get_fed_data, clean_fed_data

# %%
df_dict, df_all = preprocess()
inf_fr = get_fed_data("FRACPIALLMINMEI", units="pc1", freq="m")
df_inf, _, _ = clean_fed_data(inf_fr)

# %%
print(df_all.info())
df_all.head()

# %%
df_inf.rename(columns={"value": "inf_measured"}, inplace=True)
print(df_inf.info())
df_inf.head()

# %%
# * Combine month and year to create date
## Remove non-month entries
df_all["month"] = pd.to_numeric(df_all["month"], errors="coerce")
df_all = df_all[df_all["month"].notna()]
df_all["month"] = df_all["month"].astype(int)
df_all.dtypes

# %%

df_all["date"] = pd.to_datetime(
    dict(year=df_all["year"], month=df_all["month"], day=1), errors="coerce"
)
df_all.shape

# %%
df_combo = df_all.merge(df_inf, how="left")
print(df_combo.shape)
df_combo.head()

# %%
data = df_combo[df_combo["date"] >= "2004-01-01"][
    ["date", "inf_exp_val_inc", "inf_per_val_inc", "inf_measured"]
].melt(
    id_vars=["date"],
    var_name="Measure",
    value_vars=["inf_exp_val_inc", "inf_per_val_inc", "inf_measured"],
    value_name="Value",
)
logging.debug(data.shape)
data.dropna(inplace=True)
logging.debug(data.shape)
print(data.head)
sns.lineplot(data=data, x="date", y="Value", hue="Measure")
sns.pairplot(data=data, hue="Measure")
sns.jointplot(
    data=df_combo[["inf_per_val_inc", "inf_exp_val_inc"]],
    x="inf_per_val_inc",
    y="inf_exp_val_inc",
    kind="reg",
)
plt.show()
