import numpy as np
import sys
import os
import pandas as pd
import ast
from pathlib import Path
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning import pytorch as pl
import torch
from torch import Tensor
import io
from chemprop import data, featurizers, models, nn
from chemprop.data import BatchMolGraph
from chemprop.nn.transforms import ScaleTransform
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import r2_score as r2

def process_df(df_test, targets):
    test_smis = df_test.loc[:, "smiles"].values
    test_Ys = df_test.loc[:, targets].values
    test_data = [
        data.MoleculeDatapoint.from_smi(te_smi, te_Y)
        for te_smi, te_Y in zip(test_smis, test_Ys)
    ]
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    test_dset = data.MoleculeDataset(test_data, featurizer)
    return test_dset

props = [
    "LogD",
    "LogS",
    "Log_HLM_CLint",
    "Log_MLM_CLint",
    "Log_Caco_Papp_AB",
    "Log_Caco_ER",
    "Log_Mouse_PPB",
    "Log_Mouse_BPB",
    "Log_Mouse_MPB",
]

predictions = []
rmse_ls = []
mae_ls = []
r2_ls = []
for i in range(1, 11):
    print("##############################")
    print(f"Collecting results for fold {i}")
    
    path = f"./fold_{i}/model_files/"
    files = os.listdir(path)
    for f in files:
        if "best" in f:
            c_file = f
            
    model = models.MPNN.load_from_checkpoint(path + c_file)
    
    test = pd.read_csv(f"../data/fold_{i}/test.csv")
    test["smiles"] = test["SMILES"].copy()
    test = test.dropna(how="all", subset=props).reset_index(drop=True)
    test = test[["smiles"] + props]
    
    test_dset = process_df(test, props)
    
    print("#####################################################")
    print("Data has been loaded. Begin making predictions ...")
    num_workers = 128
    test_loader = data.build_dataloader(
        test_dset, batch_size=256, num_workers=num_workers, shuffle=False
    )
    
    with torch.inference_mode():
        trainer = pl.Trainer(
            logger=None, enable_progress_bar=True, accelerator="cpu", devices=1
        )
        test_preds = trainer.predict(model, test_loader)
    
    test_preds = torch.concat(test_preds, axis=0)
    
    names = [p + "_pred" for p in props]
    preds = pd.DataFrame(test_preds.numpy(), columns=names)
    
    df = pd.concat([test, preds], axis=1)
    predictions.append(df)
    
    mae_i = []
    rmse_i = []
    r2_i = []
    for prop in props:
        tmp = df[[prop, prop + "_pred"]].dropna()
        
        print(prop)
        mae_ = mae(tmp[prop], tmp[prop + "_pred"])
        rmse_ = rmse(tmp[prop], tmp[prop + "_pred"])
        r2_ = r2(tmp[prop], tmp[prop + "_pred"])
        print("Test MAE: ", mae_)
        print("Test RMSE: ", rmse_)
        print("Test R2: ", r2_)
        mae_i.append(mae_)
        rmse_i.append(rmse_)
        r2_i.append(r2_)
        
    mae_ls.append(mae_i)
    rmse_ls.append(rmse_i)
    r2_ls.append(r2_i)
    
df_pred = pd.concat(predictions, axis=0).reset_index(drop=True)
df_pred.to_csv(f"./cv_preds.csv", index=False)

df_mae = pd.DataFrame(mae_ls, columns=props)
df_rmse = pd.DataFrame(rmse_ls, columns=props)
df_r2 = pd.DataFrame(r2_ls, columns=props)
for prop in props:
    print(prop)
    print("Test MAE: ", df_mae[prop].mean(), " +/- ", df_mae[prop].std())
    print("Test RMSE: ", df_rmse[prop].mean(), " +/- ", df_rmse[prop].std())
    print("Test R2: ", df_r2[prop].mean(), " +/- ", df_r2[prop].std())
