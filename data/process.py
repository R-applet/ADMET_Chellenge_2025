import warnings
from io import StringIO
import os, math
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

train_df = pd.read_csv(
    "hf://datasets/openadmet/openadmet-challenge-train-data/expansion_data_train.csv"
)

d = """Assay,Log_Scale,Multiplier,Log_name
LogD,False,1,LogD
KSOL,True,1e-6,LogS
HLM CLint,True,1,Log_HLM_CLint
MLM CLint,True,1,Log_MLM_CLint
Caco-2 Permeability Papp A>B,True,1e-6,Log_Caco_Papp_AB
Caco-2 Permeability Efflux,True,1,Log_Caco_ER
MPPB,True,1,Log_Mouse_PPB
MBPB,True,1,Log_Mouse_BPB
MGMB,True,1,Log_Mouse_MPB
"""
s = StringIO(d)
conversion_df = pd.read_csv(s)
conversion_dict = dict([(x[0], x[1:]) for x in conversion_df.values])

log_train_df = train_df[["SMILES", "Molecule Name"]].copy()
for col in train_df.columns[2:]:
    log_scale, multiplier, short_name = conversion_dict[col]
    # add a new column with short_name as the column name
    log_train_df[short_name] = train_df[col].astype(float)
    if log_scale:
        # add 1 to avoid taking the log of 0
        log_train_df[short_name] = log_train_df[short_name] + 1
        # do the log transform
        log_train_df[short_name] = np.log10(log_train_df[short_name] * multiplier)

log_train_df.to_csv("log_train_data.csv",index=False)

log_train_df.sample(frac=1.0)
cv = KFold(n_splits=10, shuffle=True, random_state=42)

fold_number = 1
for train_index, test_index in cv.split(log_train_df):
    tr_df = log_train_df.iloc[train_index]
    te_df = log_train_df.iloc[test_index]
    os.mkdir(f"fold_{fold_number}")
    tr_df.to_csv(f"fold_{fold_number}/train.csv", index=False)
    te_df.to_csv(f"fold_{fold_number}/test.csv", index=False)

    fold_number += 1
