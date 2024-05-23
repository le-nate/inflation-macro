"""Constants for `process_camme.py`"""

# * Survey waves to ignore
IGNORE_SUPPLEMENTS = ["be", "cnle", "cov", "pf"]
IGNORE_HOUSING = "log"
# Only these years had separate housing surveys
IGNORE_HOUSING_YEARS = ["2016", "2017"]

# * Variables and corresponding column names (change over course of series)
# * Variables used in Andrade et al. (2023) and others of interest
VARS_DICT = {
    ## Date
    "month": {
        "2014": "MOISENQ",
    },
    ## From Andrade et al. (2023)
    "inf_exp_qual": {
        "2014": "EVOLPRIX",
    },
    "inf_exp_val": {
        "2014": ["EVPRIBAI", "EVPRIPLU"],
    },
    "consump_past": {
        "2014": "EQUIPPAS",
    },
    "consump_general": {
        "2014": "ACHATS",
    },
    ## Others
    "spend_change": {
        "2014": "DEPENSES",
    },
    "econ_exp": {
        "2014": "ECOFUT",
    },
    "personal_save_fut": {
        "2014": "ECONOMIS",
    },
    "genera_save_fut": {
        "2014": "EPARGNER",
    },
    "personal_spend_exp": {
        "2014": "EQUIPFUT",
    },
    "inf_per_qual": {
        "2014": "PRIX",
    },
    "inf_per_val": {
        "2014": ["PRIXBAIS", "PRIXPLUS"],
    },
}
