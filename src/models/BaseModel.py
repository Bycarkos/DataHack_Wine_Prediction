
import numpy as np 
import pandas as pd
from abc import ABC, abstractmethod    
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from typing import *

class BaseModel(ABC):
    
    
    def __init__(self):
        self._rmse = self.RMSE
        self._r2 = self.R_score

    def train_val_test_split(self, X, y, train_size=0.7, val_size=0.15, random_state=None):
        """
        Divide los datos en conjuntos de entrenamiento, validación y prueba.

        Args:
            X (pd.DataFrame or np.array): El conjunto de datos de características.
            y (pd.Series or np.array): El conjunto de datos de etiquetas.
            train_size (float): La proporción de los datos de entrenamiento. El valor por defecto es 0.7.
            val_size (float): La proporción de los datos de validación. El valor por defecto es 0.15.
            random_state (int): La semilla aleatoria para dividir los datos. El valor por defecto es None.

        Returns:
            Un diccionario que contiene los conjuntos de entrenamiento, validación y prueba de características y etiquetas.
        """
        assert train_size + val_size < 1, "La suma de train_size y val_size debe ser menor que 1."
        test_size = 1 - train_size - val_size

        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size/(train_size+val_size), random_state=random_state)
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def custom_kfold_partition(self, df:pd.DataFrame, target_cols:list, n_splits:int=5, shuffle:bool=True, random_state:Optional[Any]=None):
        """
        Particiones k-fold de un DataFrame basado en el valor de una o varias columnas.

        Args:
            df (pd.DataFrame): El DataFrame que se va a particionar.
            target_cols (str or list): Una cadena o lista de cadenas que contiene el nombre de las columnas en las que se basará la partición.
            n_splits (int): El número de divisiones en las que se dividirá el DataFrame.
            shuffle (bool): Si se va a barajar o no los datos antes de dividirlos en k partes.
            random_state (int): La semilla aleatoria para garantizar la reproducibilidad de los resultados.

        Returns:
            Una lista de pares de índices (train_indices, test_indices) que representan las divisiones de entrenamiento y prueba
            para cada división en k partes.
        """
        if isinstance(target_cols, str):
            target_cols = [target_cols]

        targets = df[target_cols].values
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

        result = []
        for train_index, test_index in kf.split(df):
            train_targets = df.iloc[train_index][target_cols].values
            test_targets = df.iloc[test_index][target_cols].values

            train_mask = pd.DataFrame(train_targets).isin(targets.flatten()).all(axis=1).values
            test_mask = pd.DataFrame(test_targets).isin(targets.flatten()).all(axis=1).values

            result.append((train_index[train_mask], test_index[test_mask]))

        return result
    
    
    def RMSE(self, prediction:np.ndarray, target:np.ndarray):
        mse = mean_squared_error(target, prediction)
        return (mse)**0.5
    
    
    def R_score(self,prediction:np.ndarray, target: np.ndarray):
        return r2_score(target, prediction)
    
    
    @abstractmethod
    def trainer(self):
        raise NotImplementedError
    
    @abstractmethod
    def validator(self):
        raise NotImplementedError






