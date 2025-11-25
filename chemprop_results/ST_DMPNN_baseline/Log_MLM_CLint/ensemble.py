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

def process_df(df_test):
    test_smis = df_test.loc[:, "smiles"].values
    test_data = [
        data.MoleculeDatapoint.from_smi(te_smi)
        for te_smi in test_smis
    ]
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    test_dset = data.MoleculeDataset(test_data, featurizer)
    return test_dset

prop = "Log_MLM_CLint"

#df = pd.read_csv("../data/log_train_data.csv")
df = pd.read_csv(
        "hf://datasets/openadmet/openadmet-challenge-test-data-blinded/expansion_data_test_blinded.csv"
)
df["smiles"] = df["SMILES"].copy()
data_dset = process_df(df)
num_workers=15
data_loader = data.build_dataloader(                                                                                                                    
    data_dset, batch_size=256, num_workers=num_workers, shuffle=False                                            
)

predictions = []
for i in range(1,11):
    path = f"./fold_{i}/model_files/"
    files = os.listdir(path)
    for f in files:
        if "best" in f:
            c_file = f
            
    model = models.MPNN.load_from_checkpoint(path + c_file)
    with torch.inference_mode():
        trainer = pl.Trainer(
            logger=None, enable_progress_bar=True, accelerator="auto", devices=1
        )
        data_preds = trainer.predict(model, data_loader)
    data_preds = torch.concat(data_preds, axis=0)
    predictions.append(pd.DataFrame(data_preds, columns=[prop]))

avg_predictions = sum(predictions)/len(predictions)
df_data = pd.concat([df[["SMILES","Molecule Name"]],avg_predictions],axis=1)
#df_data.to_csv("train_ensemble_preds.csv",index=False)
df_data.to_csv("test_ensemble_preds.csv",index=False)
