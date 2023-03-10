import os
from pathlib import Path
from typing import Optional

import pandas as pd
from models.Xgboost import XGboost
import numpy as np

from utils import save_pickle, get_best_model, write_response_file
from hydra.utils import get_original_cwd, to_absolute_path, instantiate
import hydra
from omegaconf import DictConfig, OmegaConf


import warnings
warnings.filterwarnings("ignore")

@hydra.main(config_path="./configs", config_name="train", version_base="1.1")
def main(cfg:DictConfig):
    print(cfg)
    # TODO CANVIAR TOT AIXÔ PER UTILITZAR EL HYDRA
    name_file_loaded = cfg.data.utils.name
    path_data = cfg.data.utils.path

    print("Loading data...")
    df = instantiate(cfg.data.load, filepath_or_buffer=os.path.join(path_data , name_file_loaded))
    del df["ALTITUD"]
    print("Done!")

    # Create an instance of YourModelClass
    params = cfg.models.model
    export_dir = os.path.join(cfg.outputs.exports.path,cfg.outputs.exports.which)  ## which mean kfold or static cross val (random_split)
    
    print("Loading model...")
    model = XGboost(params, export_dir)
    print("Done!")

    # TRAIN BLOCK
    print("*** TRAIN BLOCK ***")
    print("Preparing splits...")
    df_train = df.loc[df["CAMPAÑA"] != 22]
    df_test = df.loc[df["CAMPAÑA"] == 22]
    
    n_splits = len(df_train["CAMPAÑA"].unique())
    
    folds = None #model.custom_kfold_partition(df_train, target_cols="CAMPAÑA", n_splits=n_splits, random_state=1)

    x_train = df_train.loc[:, df_train.columns != "PRODUCCION"]
    y_train = df_train.loc[:, "PRODUCCION"]
    print("Done!")

    print("Staring training...")
    model.train(x=x_train, y=y_train, kfold_indexes=folds, write_compare=cfg.setup.val_compare_file)
    print("Done!")

    checkpoint_dir = cfg.outputs.checkpoints.path
    if checkpoint_dir is not None:
        print("Saving checkpoint...")
        if not os.path.exists(checkpoint_dir):
            os.mkdirs(checkpoint_dir)
        model_name = os.path.join(checkpoint_dir, cfg.outputs.checkpoints.name_file)
        save_pickle(model, model_name)
        print("Done!")
    print("*** END TRAIN BLOCK ***")

    # TEST BLOCK
    print("*** TEST BLOCK ***")
    x_test = df_test.loc[:, df_test.columns != "PRODUCCION"]
    
    if cfg.setup.kfold_agg:
        print("Using aggregation of kfolds...")
        to_agg = np.zeros(x_test.shape[0])
        for dicts, _ in model._dict_results: 
            model_to_test = dicts["model"]
            _, test_predictions = model.validator(model=model_to_test, x_val=x_test)
            to_agg = np.add(to_agg, np.array(test_predictions))
        
        test_predictions = np.divide(to_agg, n_splits)
        print("Done!")

    else:
        print("Using best model...")
        best_model_idx = get_best_model(model._validations_results)
        model_to_test = model._dict_results[best_model_idx][0]["model"]    
        _, test_predictions = model.validator(model=model_to_test, x_val=x_test)
        print("Done!")

    print("Writing response...")
    export_name = cfg.outputs.exports.name_file
    name_file_response = export_dir + "/UH2023_Universitat Autònoma de Barcelona (UAB)_AskGPC_1.csv"
    write_response_file(x_test, test_predictions, name_file_response)
    print("Done!")


if __name__ == "__main__":
    main()
    
    # TODO CANVIAR TOT AIXÔ PER UTILITZAR EL HYDRA
    exit()
    """
    Original train set: UH_2023_TRAIN.txt
    New processed train set: 
        - DATA_TRAIN_JOINED_DAYS.csv, aggregated by days
        - DATA_TRAIN_JOINED_MONTHS.csv, aggregated by months
    """

    main(input_file=input_file, model_config=model_config, checkpoint_dir=checkpoint_dir, export_dir=export_dir,
         use_aggregation_of_kfolds=use_aggregation_of_kfolds, write_compare=write_compare)
    
   