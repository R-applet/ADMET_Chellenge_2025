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
import matplotlib.pyplot as plt

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

target = "LogS"

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
    
    test = pd.read_csv(f"../../data/fold_{i}/test.csv")
    test["smiles"] = test["SMILES"].copy()
    test = test.dropna(how="all", subset=[target]).reset_index(drop=True)
    test = test[["smiles",target]]
    
    test_dset = process_df(test, [target])
    
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
    
    preds = pd.DataFrame(test_preds.numpy(), columns=[target+"_pred"])
    
    df = pd.concat([test, preds], axis=1).reset_index(drop=True)
    predictions.append(df)
    
    mae_i = mae(df[target],df[target+"_pred"])
    rmse_i = rmse(df[target],df[target+"_pred"])
    r2_i = r2(df[target],df[target+"_pred"])
        
    print("Test MAE:", mae_i)
    print("Test RMSE:", rmse_i)
    print("Test R2:", r2_i)

    mae_ls.append(mae_i)
    rmse_ls.append(rmse_i)
    r2_ls.append(r2_i)
    
df_pred = pd.concat(predictions, axis=0).reset_index(drop=True)
df_pred.to_csv(f"./cv_preds.csv", index=False)

print("Test MAE: ", np.mean(mae_ls), " +/- ", np.std(mae_ls))
print("Test RMSE: ", np.mean(rmse_ls), " +/- ", np.std(rmse_ls))
print("Test R2: ", np.mean(r2_ls), " +/- ", np.std(r2_ls))

mae = mae(df_pred[target], df_pred[target+"_pred"])
rmse = rmse(df_pred[target], df_pred[target+"_pred"])
r2 = r2(df_pred[target], df_pred[target+"_pred"])

x = np.linspace(min(df_pred[target].min(), df_pred[target+"_pred"].min()), max(df_pred[target].max(), df_pred[target+"_pred"].max()), 100)
plt.figure(figsize=(6,6))
plt.plot(df_pred[target], df_pred[target+"_pred"], "co", mew=0.0, alpha=0.4)
plt.plot(x, x, "k--")
plt.xlabel(f"Measured {target}",fontsize=14)
plt.ylabel(f"Predicted {target}",fontsize=14)
plt.title(f"MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}",fontsize=12)
plt.tick_params(labelsize=12)
plt.savefig("parity.png", dpi=666)
