"""Analysis of CAMME data"""

# %%
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.process_camme import preprocess
from src.retrieve_data import get_fed_data, clean_fed_data
from src.utils.logging_helpers import define_other_module_log_level

# * Logging settings
logger = logging.getLogger(__name__)
define_other_module_log_level("warning")
logger.setLevel(logging.DEBUG)

# * Get data directory folder
parent_dir = Path(__file__).parents[1]
camme_dir = parent_dir / "data" / "camme"


# %%
df_dict, df_all = preprocess(camme_dir)
# inf_fr = get_fed_data("FRACPIALLMINMEI", units="pc1", freq="m")
# df_inf, _, _ = clean_fed_data(inf_fr)

# %%
print(df_all.info())
df_all.head()

# %%
# df_inf.rename(columns={"value": "inf_measured"}, inplace=True)
# print(df_inf.info())
# df_inf.head()

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
# df_combo = df_all.merge(df_inf, how="left")
# print(df_combo.shape)
# df_combo.head()
df_combo = df_all.copy()

# %%
data = df_combo[df_combo["date"] >= "2004-01-01"][
    ["date", "inf_exp_val_inc", "inf_per_val_inc"]
].melt(
    id_vars=["date"],
    var_name="Measure",
    value_vars=["inf_exp_val_inc", "inf_per_val_inc"],
    value_name="Value",
)
logging.debug(data.shape)
data.dropna(inplace=True)
logging.debug(data.shape)
print(data.head)

# %%
sns.lineplot(data=data, x="date", y="Value", hue="Measure")
sns.pairplot(data=data)
sns.jointplot(
    data=df_combo[["inf_per_val_inc", "inf_exp_val_inc"]],
    x="inf_per_val_inc",
    y="inf_exp_val_inc",
    kind="reg",
)

# %%

x = data[data["Measure"] == "inf_per_val_inc"]["Value"].to_numpy()

sns.kdeplot(data=data, x="Value", hue="Measure")
plt.yscale("log")
plt.xscale("log")

hist, bin_edges = np.histogram(x, bins=50, density=False)
bin_center = (bin_edges[:-1] + bin_edges[1:]) / 2

plt.figure()
plt.hist(x, bins=50, density=False)
plt.errorbar(bin_center, hist, yerr=50, fmt=".")

plt.show()
