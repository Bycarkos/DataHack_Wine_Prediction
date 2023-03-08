import os
from pathlib import Path
from typing import Optional

import pandas as pd
from models.Xgboost import XGboost
import numpy as np

from src.utils import save_pickle, get_best_model, write_response_file

import warnings
warnings.filterwarnings("ignore")


def main(input_file: Path, model_config: dict, checkpoint_dir: Optional[Path] = None, export_dir: Optional[Path] = None,
         use_aggregation_of_kfolds: bool = False, write_compare: bool = False):

    # Read CSV file into a pandas DataFrame
    print("Loading data...")
    df = pd.read_csv(input_file)
    print("Done!")

    # Create an instance of YourModelClass
    print("Loading model...")
    model = XGboost(model_config, export_dir)
    print("Done!")

    # TRAIN BLOCK
    print("*** TRAIN BLOCK ***")
    print("Preparing splits...")
    df_train = df.loc[df["CAMPAÑA"] != 22]
    df_test = df.loc[df["CAMPAÑA"] == 22]
    
    n_splits = len(df_train["CAMPAÑA"].unique())
    
    folds = None  # model.custom_kfold_partition(df_train, target_cols="CAMPAÑA", n_splits=n_splits, random_state=1)

    x_train = df_train.loc[:, df_train.columns != "PRODUCCION"]
    y_train = df_train.loc[:, "PRODUCCION"]
    print("Done!")

    print("Staring training...")
    model.train(x=x_train, y=y_train, kfold_indexes=folds, write_compare=write_compare, model_config=model_config)
    print("Done!")

    if checkpoint_dir:
        print("Saving checkpoint...")
        checkpoint_dir.mkdir(exist_ok=True)
        model_name = checkpoint_dir / "latest.pkl"
        save_pickle(model, model_name)
        print("Done!")
    print("*** END TRAIN BLOCK ***")

    # TEST BLOCK
    print("*** TEST BLOCK ***")
    x_test = df_test.loc[:, df_test.columns != "PRODUCCION"]
    
    if use_aggregation_of_kfolds:
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
    name_file_response = export_dir / "UH2023_Universitat Autònoma de Barcelona (UAB)_AskGPC_1.csv"
    write_response_file(x_test, test_predictions, name_file_response)
    print("Done!")


if __name__ == "__main__":
    """
    Original train set: UH_2023_TRAIN.txt
    New processed train set: 
        - DATA_TRAIN_JOINED_DAYS.csv, aggregated by days
        - DATA_TRAIN_JOINED_MONTHS.csv, aggregated by months
    """

    # Constants
    ROOT = Path(os.getcwd()).absolute()
    ROOT_DATA = ROOT.parent / "data"
    ROOT_OUTPUTS = ROOT.parent / "outputs"
    ROOT_OUTPUTS.mkdir(exist_ok=True)

    # Inputs
    input_file = ROOT_DATA / "DATA_TRAIN_JOINED_DAYS_NO_ALT.csv"

    # Outputs
    checkpoint_dir = ROOT_OUTPUTS / "checkpoints"
    export_dir = ROOT_OUTPUTS / "export"
    checkpoint_dir.mkdir(exist_ok=True)
    export_dir.mkdir(exist_ok=True)

    # Model configuration
    model_config = {
        "n_estimators": 2000,
        "n_jobs": -1,
        "max_depth": 7,
        "eta": 0.1,
        "subsample": 0.7,
        "colsample_bytree": 0.8,
        'objective': 'reg:squarederror',
        'tree_method': 'gpu_hist',
    }

    use_aggregation_of_kfolds = True
    write_compare = True

    main(input_file=input_file, model_config=model_config, checkpoint_dir=checkpoint_dir, export_dir=export_dir,
         use_aggregation_of_kfolds=use_aggregation_of_kfolds, write_compare=write_compare)
    
   