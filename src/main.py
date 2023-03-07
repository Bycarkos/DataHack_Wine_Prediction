import pandas as pd
from models.Xgboost import XGboost
import numpy as np
from utils import *


def main(csv_file_path, name_file_to_save_model:Optional[Path]=None, name_file_response:Optional[Path]=None, **config:dict):
    config_agg_folds = config.get("agg_folds", None)
    config.pop("agg_folds", None)
    
    config_write_compare = name_file_model.split("/")[-1] if config.get("compare", None) is not None else None
    config.pop("compare", None)
    
    # Read CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)#, sep="|")
    #del df["ALTITUD"]
    # Create an instance of YourModelClass
    model = XGboost(config)

    # TRAIN BLOCK    
    df_train = df.loc[df["CAMPAÑA"] != 22]
    df_test = df.loc[df["CAMPAÑA"] == 22]
    
    n_splits = len(df_train["CAMPAÑA"].unique())
    
    folds = model.custom_kfold_partition(df_train, target_cols = "CAMPAÑA", n_splits=n_splits, random_state=1)
    
    X_train = df_train.loc[:,df_train.columns != "PRODUCCION"]
    y_train = df_train.loc[:,"PRODUCCION"]
        
    # SAVE MODEL IF NEEDED
    if name_file_to_save_model is not None:
        if (not os.path.exists(name_file_to_save_model)):
            model.train(X=X_train, y=y_train, kfold_indexes=folds,write_compare_name=config_write_compare, **config)
            set_pickle(model, name_file_to_save_model)
            
        else:
            model = get_pickle(name_file_to_save_model)
                
    else:
        model.train(X=X_train, y=y_train, kfold_indexes=folds, **config)
        



    
    
    # TEST BLOCK
    X_test = df_test.loc[:,df_test.columns != "PRODUCCION"]
    
    # AQUEST IF ÉS PER ELEGIR SI VOLS AGAR LA MITJA DE TOTS ELS MODELS O NOMÉS VOLS AGAFAR ES MILLOR DE QUE HA FET SA VALIDACIÓ
    if config_agg_folds is not None:
        to_agg = np.zeros(X_test.shape[0])
        for dicts, _ in model._dict_results: 
            model_to_test = dicts["model"]
            _, test_predictions = model.validator(model=model_to_test, X_val=X_test)
            to_agg = np.add(to_agg, np.array(test_predictions))
        
        test_predictions = np.divide(to_agg, n_splits)
    
    else:
        
        best_model_idx = get_best_model(model._validations_results)
        model_to_test = model._dict_results[best_model_idx][0]["model"]    
        _, test_predictions = model.validator(model=model_to_test, X_val=X_test)
        
    write_response(X_test, test_predictions, name_file_response)
         
if __name__ == "__main__":

    #UH_2023_TRAIN.txt
    #DATA_TRAIN_JOINED_DAYS.csv
    name_file_model = os.path.join(ROOT,"models_save", "TRAINED_JOINED_DAYS")
    name_file_response = os.path.join(ROOT,"results", "TRAINED_JOINED_DAYS.txt")
    
    
    
    config = {
        "n_estimators":1000,
        "n_jobs":-1 ,
        "max_depth": 7,
        "eta":0.1, 
        "subsample":0.7,
        "colsample_bytree":0.8,
        'objective': 'reg:squarederror',
        'tree_method': 'gpu_hist',
        "compare":True # Comment to dissable
        #"agg_folds": True
    }
    
    # more flags to add:

    
    main('data/DATA_TRAIN_JOINED_DAYS.csv',name_file_to_save_model=name_file_model,name_file_response=name_file_response, **config)
    
   