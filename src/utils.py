import pickle
from pathlib import Path
import os
from typing import *
import pandas as pd
global ROOT

ROOT = os.getcwd()

def set_pickle(object_to_save:object, path:Path):
    with open(path, "wb") as handle:
        pickle.dump(object_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)    

def get_pickle(path:Path):
    with open(path, "rb") as handle:
        o=pickle.load(handle)     
    return o


def get_best_model(results:List[Tuple[float, float]]):
    return results.index(min(results, key = lambda t: t[0]["rmse"]))
    
    
def write_compare(X:pd.DataFrame, predictions:list, gt:list, name_file:str, fold:str):
    X.loc[:,"RESPONSE"] = predictions
    X.loc[:,"REAL"] = gt
    
    
    to_save= X.loc[:,["ID_FINCA", "VARIEDAD", "REAL", "RESPONSE"]]
    save_folder = os.path.join(ROOT, "results", "kfolds", name_file)
    path_save = os.path.join(save_folder, name_file+f"_{fold}"+".txt")
    if (not os.path.exists(save_folder)):
        os.mkdir(save_folder)
    
    print("LA PRODUCCIÓ FINAL VAL= {}".format(to_save["RESPONSE"].sum()))    
    print("LA PRODUCCIÓ FINAL REAL= {}".format(to_save["REAL"].sum()))
    print("LA DIFERÈNCIA = {}".format(to_save["RESPONSE"].sum()-to_save["REAL"].sum()))
        

    to_save.to_csv(path_save,sep="|", index=False, header=False)
    
def write_response(X_df:pd.DataFrame, predictions:list, name_file:Path):
    X_df.loc[:,"RESPONSE"] = predictions
    
    to_save = X_df.groupby(["ID_FINCA", "VARIEDAD", "MODO", "TIPO", "COLOR", "SUPERFICIE"]).sum().reset_index()
    
    columns_to_save = ["ID_FINCA", "VARIEDAD", "MODO", "TIPO", "COLOR", "SUPERFICIE", "RESPONSE"] 
    to_save = to_save[columns_to_save].sort_values(columns_to_save, ascending=[True,True,True,True,True,True, True])
    to_save.to_csv(name_file, index=False, header=False, sep="|")    
    print("La PRODUCCIÓ FINAL ÉS DE {}".format(to_save["RESPONSE"].sum()))    
    
if __name__ == "__main__":
    a =get_best_model([(5387.542541484709, 0.8262704714764487), (5653.556557145037, 0.7874868609077237), (6386.709133959345, 0.7660141609064283), (6788.611177296115, 0.7717278183969589), (7180.175636133389, 0.7229620085382913)])
    print(a)