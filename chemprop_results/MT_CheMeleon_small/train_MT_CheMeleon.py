import numpy as np
import sys
import os
import pickle
import pandas as pd
import ast
from pathlib import Path
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning import pytorch as pl
import torch
import io
from chemprop import data, featurizers, models, nn

hp_dict = {
    "ffn_num_layers": 1,
    "ffn_hidden_dim": 300,
    "dropout": 0.10,
    "warmup_epochs": 5
}

def process_df(df_train, df_test, targets):
    train_smis = df_train.loc[:, "smiles"].values
    train_Ys = df_train.loc[:, targets].values
    test_smis = df_test.loc[:, "smiles"].values
    test_Ys = df_test.loc[:, targets].values

    test_data = [
        data.MoleculeDatapoint.from_smi(te_smi, te_Y)
        for te_smi, te_Y in zip(test_smis, test_Ys)
    ]
    tmp_data = [
        data.MoleculeDatapoint.from_smi(tr_smi, tr_Y)
        for tr_smi, tr_Y in zip(train_smis, train_Ys)
    ]
    tmp_mols = [t.mol for t in tmp_data]

    split_idx = data.make_split_indices(tmp_mols, "random", (0.9, 0, 0.1))

    train_data, dummy_data, val_data = data.split_data_by_indices(
        tmp_data,
        train_indices=split_idx[0],
        val_indices=None,
        test_indices=split_idx[2],
    )

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_dset = data.MoleculeDataset(train_data[0], featurizer)
    scaler = train_dset.normalize_targets()
    val_dset = data.MoleculeDataset(val_data[0], featurizer)
    val_dset.normalize_targets(scaler)
    test_dset = data.MoleculeDataset(test_data, featurizer)

    return train_dset, val_dset, test_dset, scaler


def chemeleon_multitask_train(train_dset, val_dset, test_dset, targets, scaler, hp_dict):
    num_workers = 16
    train_loader = data.build_dataloader(
        train_dset, batch_size=hp_dict["batch_size"], num_workers=num_workers
    )
    val_loader = data.build_dataloader(
        val_dset,
        batch_size=hp_dict["batch_size"],
        num_workers=num_workers,
        shuffle=False,
    )
    test_loader = data.build_dataloader(
        test_dset,
        batch_size=hp_dict["batch_size"],
        num_workers=num_workers,
        shuffle=False,
    )

    chemeleon_mp = torch.load("path/to/chemeleon_mp.pt", weights_only=True)
    mp = nn.BondMessagePassing(**chemeleon_mp['hyper_parameters'])
    mp.load_state_dict(chemeleon_mp['state_dict'])
    agg = nn.NormAggregation()
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    ffn = nn.RegressionFFN(
        n_tasks=len(targets),
        task_weights=torch.as_tensor(hp_dict["task_weights"]),
        output_transform=output_transform,
        input_dim=mp.output_dim,
        hidden_dim=hp_dict["ffn_hidden_dim"],
        n_layers=hp_dict["ffn_num_layers"],
        dropout=hp_dict["dropout"]
    )
    batch_norm = hp_dict["batch_norm"]
    metric_list = [nn.metrics.RMSE(), nn.metrics.MAE(), nn.metrics.R2Score()]
    mpnn = models.MPNN(mp, agg, ffn, batch_norm, metric_list, warmup_epochs=hp_dict["warmup_epochs"])

    checkpointing = ModelCheckpoint(
        dirpath="model_files",
        filename="best-{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_last=True,
    )
    earlystopping = EarlyStopping(
        monitor="val_loss", patience=hp_dict["patience"], mode="min"
    )

    trainer = pl.Trainer(
        logger=True,
        enable_checkpointing=True,  # Use `True` if you want to save model checkpoints. The checkpoints will be saved in the `checkpoints` folder.
        enable_progress_bar=True,
        accelerator="auto",
        devices=1,
        max_epochs=hp_dict["max_epochs"],  # number of epochs to train for  #HP
        callbacks=[checkpointing, earlystopping],
    )

    trainer.fit(mpnn, train_loader, val_loader)
    results = trainer.test(mpnn, test_loader)

    return trainer, mpnn, results

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

w_tasks = [1.018, 1.000, 1.364, 1.134, 2.377, 2.373, 3.939, 5.259, 23.099]

train = pd.read_csv(f"../../data/fold_1/train.csv")
test = pd.read_csv(f"../../data/fold_1/test.csv")

hp_dict["max_epochs"] = 400
hp_dict["patience"] = 40
hp_dict["batch_norm"] = True
hp_dict["batch_size"] = 16
hp_dict["task_weights"] = w_tasks

train["smiles"] = train["SMILES"].copy()
test["smiles"] = test["SMILES"].copy()

train = train.dropna(how="all", subset=props).reset_index(drop=True)
train = train[["smiles"] + props]
test = test.dropna(how="all", subset=props).reset_index(drop=True)
test = test[["smiles"] + props]
print("#####################################################")
train_dset, val_dset, test_dset, scaler = process_df(train, test, props)
print(f"#### BEGINNING MODEL TRAINING/TESTING ####")
trainer_, mpnn_, results_ = chemeleon_multitask_train(
    train_dset, val_dset, test_dset, props, scaler, hp_dict
)
