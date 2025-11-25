import pandas as pd

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

df_ = pd.read_csv(
    "hf://datasets/openadmet/openadmet-challenge-test-data-blinded/expansion_data_test_blinded.csv"
)

df_ls = [pd.read_csv(f"{prop}/test_ensemble_preds.csv")[[prop]] for prop in props]

df = pd.concat([df_]+df_ls, axis=1)

df.to_csv("test_ensemble_preds.csv",index=False)
