import os
from pathlib import Path

import numpy as np

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from models.Xgboost import XGboost
from data.Collector import Collector
from data.DataLoader import DataLoader

from utils import *

import warnings
warnings.filterwarnings("ignore")


@hydra.main(config_path="./configs", config_name="train", version_base="1.1")
def main(cfg: DictConfig):
    print(cfg)
    print("Loading DataLoader...")
    dl = DataLoader(cfg.data.DataLoader)    
    df = dl._data
    del df["ALTITUD"]
    print("Done!")
    
    # Create an instance of YourModelClass
    params = cfg.models.model
    export_dir = Path(os.path.join(cfg.outputs.exports.path, cfg.outputs.exports.which))  ## which mean kfold or static cross val (random_split)
    
    
    # CANVIAR AIXÔ PER A QUE SIGUI un INSTANTIATE
    print("Loading model...")
    model = instantiate(cfg.models.model, export_dir=export_dir)
    print("Done!")
    # TRAIN BLOCK
    print("*** TRAIN BLOCK ***")
    df_train = df.loc[df["CAMPAÑA"] != 22]
    df_test = df.loc[df["CAMPAÑA"] == 22]
    
    x_train = df_train.loc[:, df_train.columns != "PRODUCCION"]
    y_train = df_train.loc[:, "PRODUCCION"]
    
    print("PREPARING PARTITIONS AND TRAINING THE MODELS")
    
    if cfg.setup.mode == "kfold":
        n_splits = len(df_train["CAMPAÑA"].unique())

        train_folds_idx, batches_train, test_folds_idx, batches_test =  dl.kfold_collector(x_train,n_splits=n_splits, columns="CAMPAÑA")
        
        best_model = model.train(x_train=x_train, y_train=y_train, kfold_indexes=[train_folds_idx, test_folds_idx] ,write_compare=cfg.setup.val_compare_file)

    else:
        train_pack, val_pack, test_pack = dl.train_val_test_split_collector(x_train, y_train, metaclasses=["CAMPAÑA"], ret="dataframe")
        x_train, y_train, train_batches = train_pack
        x_val, y_val, val_batches = val_pack
        x_test, y_test, test_batches = test_pack    
        best_model = model.train(x_train = x_train, y_train = y_train, x_val=x_val, y_val=y_val ,write_compare=cfg.setup.val_compare_file)
        print("*** TEST BLOCK ***")
        model.validator(model=best_model, x_test=x_test, y_test=y_test) # Es pot afegir write compare

    print("Done!")

    checkpoint_dir = cfg.outputs.checkpoints.path
    if checkpoint_dir is not None:
        print("Saving checkpoint...")
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        model_name = Path(os.path.join(checkpoint_dir, cfg.outputs.checkpoints.name_file))
        save_pickle(model, model_name)
        print("Done!")
    print("*** END TRAIN BLOCK ***")

    # TEST BLOCK
    print("*** TEST BLOCK ***")
    x_test = df_test.loc[:, df_test.columns != "PRODUCCION"]
    print("Using best model...")  
    _, test_predictions = model.validator(model=best_model, x_test=x_test)
    print("Done!")

    print("Writing response...")
    export_name = export_dir / cfg.outputs.exports.name_file
    write_response_file(x_test, test_predictions, export_name)
    print("Done!")


if __name__ == "__main__":
    """
    Original train set: UH_2023_TRAIN.txt
    New processed train set: 
        - DATA_TRAIN_JOINED_DAYS.csv, aggregated by days
        - DATA_TRAIN_JOINED_MONTHS.csv, aggregated by months
    """

    main()
