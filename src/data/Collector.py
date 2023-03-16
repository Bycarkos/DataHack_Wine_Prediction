import torch 
import numpy as np 
import pandas as pd 
import torch.nn as nn
import torch.functional as F
import random


from typing import *
from pathlib import *


from sklearn.model_selection import KFold, train_test_split
from math import ceil

class Collector():
    
    
    def __init__(self, batch_size:int, shuffle:bool, random_state:Any) -> None:
        
        ## utils for the collector
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._random_state = random_state
                        
        
    def get_k_folds(self, df:pd.DataFrame, n_splits:int, columns:List[str]):
        targets = df[columns].values
        kf = KFold(n_splits=n_splits, shuffle=self._shuffle, random_state=self._random_state)
        
        train_folds = []
        test_folds = []
        for train_index, test_index in kf.split(df):
            train_targets = df.iloc[train_index][columns].values
            test_targets = df.iloc[test_index][columns].values

            train_mask = pd.DataFrame(train_targets).isin(targets.flatten()).all(axis=1).values
            test_mask = pd.DataFrame(test_targets).isin(targets.flatten()).all(axis=1).values

            train_folds.append(train_index[train_mask])
            test_folds.append(test_index[test_mask])
            
        return np.array(train_folds), np.array(test_folds)
    
    def get_train_val_test_split(self, X,y, sizes:Tuple[float, float, float]= [0.8,0.1,0.1], metaclasses: Optional[Tuple[str, str]]=None, ret="dataframe") -> Tuple[torch.Tensor, ...]:
        
        assert sum(sizes) == 1., "Error with the proportions"
        if metaclasses is not None:
            X["target"] = y
            metatrain_prop, metaval_prop, metatest_prop = float(sizes[0]), float(sizes[1]), float(sizes[2])
            ngroups = len(metaclasses)
            
            grouped_pandas = X.groupby(metaclasses)
            
            train = pd.DataFrame()
            validation = pd.DataFrame()
            test = pd.DataFrame()
            
            
            groups = list(grouped_pandas.groups.keys())
      
            train_groups = random.sample(groups, round(metatrain_prop * len(groups)))
            collect_groups = list(set(groups)- set(train_groups))      
            val_groups = random.sample(collect_groups, ceil((metaval_prop)* len(groups)))
            test_groups = list(set(groups)- set(train_groups) - set(val_groups))

            for name, group in grouped_pandas:

                if bool(set([name])&set(train_groups)) == True:             
                    train = pd.concat([train,group])
                    
                elif bool(set([name]) & set(val_groups)) == True:
                    validation = pd.concat([validation,group])
                
                elif bool(set([name])&set(test_groups)) == True:
                    test = pd.concat([test,group])
            
                    
            df_train = train.reset_index().drop("index", axis=1)
            df_validation = validation.reset_index().drop("index", axis=1)
            df_test = test.reset_index().drop("index", axis=1)

            ### Train
            y_train = (df_train["target"])
            X_train = (df_train.drop("target", axis=1))
            
            ## validation
            y_val = (df_validation["target"].values)
            X_val = (df_validation.drop("target", axis=1))          
            
            ##test
            y_test = (df_test["target"])
            X_test = (df_test.drop("target", axis=1))

        
        else:
            train_size, val_size = sizes[0], sizes[1]
            test_size = 1 - train_size - val_size

            x_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=self._random_state)
            X_train, X_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=val_size/(train_size+val_size), random_state=self._random_state)
                    
        if ret != "dataframe":
            print("HOLA HOLA")
            return torch.tensor(X_train.values) , torch.tensor(y_train.values), torch.tensor(X_val.values), torch.tensor(y_val.values), torch.tensor(X_test.values), torch.tensor(y_test.values)
        else:
    
            return X_train , y_train, X_val, y_val, X_test, y_test
        
    @staticmethod
    def get_batch_idx(size, batch_size):    
        nb_batch = int(np.ceil(size / float(batch_size)))
        res = [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(0, nb_batch)]
        return res
        