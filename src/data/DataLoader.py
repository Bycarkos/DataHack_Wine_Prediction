import torch 
import numpy as np 
import pandas as pd 
import torch.nn as nn
import torch.functional as F
import os


from typing import *
from pathlib import *


from hydra.utils import instantiate
from data.Collector import Collector
from sklearn.model_selection import KFold, train_test_split


class DataLoader():
    
    
    def __init__(self, cfg) -> None:
        
        ## Dataset
        self._filepath = os.path.join(cfg.utils.path, cfg.utils.name)
        self._data = instantiate(cfg.load, filepath_or_buffer=self._filepath)
                
        self._collector = instantiate(cfg.collector)
        
    def kfold_collector(self, df:pd.DataFrame, n_splits:int, columns:List[str]):
        train_folds, test_folds = self._collector.get_k_folds(df=df, n_splits=n_splits, columns=columns)
         
        batches_train_fold = []
        batches_test_fold = []
        
        for ftr, fte in zip(train_folds, test_folds):
            batches_train_fold.append(self._collector.get_batch_idx(ftr.shape[0], self._collector._batch_size))
            batches_test_fold.append(self._collector.get_batch_idx(fte.shape[0], self._collector._batch_size))
        
        return train_folds, np.array(batches_train_fold), test_folds, np.array(batches_test_fold)
        
    def train_val_test_split_collector(self, X:pd.DataFrame, y:pd.DataFrame, sizes: Optional[Tuple[float, float, float]] = [0.8,0.1,0.1],metaclasses: Optional[Tuple[str, str]]=None, ret="dataframe"):
        x_train, y_train, x_val, y_val, x_test, y_test = self._collector.get_train_val_test_split(X=X, y=y,sizes=sizes, metaclasses=metaclasses, ret=ret )

        batches_train = self._collector.get_batch_idx(x_train.shape[0], self._collector._batch_size)
        batches_val = self._collector.get_batch_idx(x_val.shape[0], self._collector._batch_size)
        batches_test = self._collector.get_batch_idx(x_test.shape[0], self._collector._batch_size)
        
        return (x_train, y_train, batches_train), (x_val, y_val, batches_val), (x_test, y_test, batches_test)
                   
        
    def reset_collector(self, cfg):
        self._collector = Collector(cfg=cfg)
    
    
    def __getitem__(self, index):
        return self._data.iloc[index]
    
    def __setitem__(self, index, value):
        assert len(index) == len(value), "Mismatch of shapes"
        self._data.iloc[index] = value
        
        
    
    @staticmethod
    def load(path:Path, **params):
        return pd.read_csv(filepath_or_buffer=path, **params)

        