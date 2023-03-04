import numpy as np 
import pandas as pd 
from BaseModel import BaseModel
import xgboost as xgb
from typing import *

class XGboost(BaseModel):
    
    def __init__(self, cfg:dict):
        super().__init__()
        
        
    def initialize_matrix(self,X:np.ndarray,y:np.ndarray):
        return xgb.DMatrix(X, label=y)
    
    def validator(self,model, X_val, y_val):
        """
        Valida un modelo de regresión XGBoost y devuelve la métrica de validación RMSE.

        Args:
            model (xgb.XGBRegressor): El modelo de regresión XGBoost entrenado.
            X_val (pd.DataFrame or np.array): El conjunto de validación de características.
            y_val (pd.Series or np.array): El conjunto de validación de etiquetas.

        Returns:
            La métrica de validación (RMSE).
        """
        y_pred = model.predict(X_val)

        rmse = self._rmse(y_val, y_pred)

        return rmse
            
    def trainer(self,X_train:np.ndarray, y_train:np.ndarray, params=None):
        """
        Entrena un modelo de regresión XGBoost y devuelve el modelo entrenado, RMSE, número de muestras y número de características.

        Args:
            X_train (pd.DataFrame or np.array): El conjunto de entrenamiento de características.
            y_train (pd.Series or np.array): El conjunto de entrenamiento de etiquetas.
            params (dict): Los parámetros del modelo de regresión XGBoost.

        Returns:
            Un diccionario que contiene el modelo de regresión XGBoost entrenado, RMSE, número de muestras y número de características.
        """
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        num_samples, num_features = X_train.shape

        model = xgb.XGBRegressor(**params)
        model.fit(dtrain, y_train)

        y_pred = model.predict(X_train)
        rmse = self._rmse(y_train, y_pred)

        result = {
            'model': model,
            'rmse': rmse,
            'num_samples': num_samples,
            'num_features': num_features
        }

        return result
    
    
    def train(self, X:np.ndarray, y:np.ndarray, kfold_indexes:Optional[None], **params):
        
        """
        Entrena un modelo de regresión XGBoost y devuelve tres listas con la prediccion según la métrica RMSE.

        Args:
            X (pd.DataFrame or np.array): Todo el conjunto de datos
            y_train (pd.Series or np.array): El conjunto de etiquetas.
            params (dict): Los parámetros del modelo de regresión XGBoost.

        Returns:
            Tres listas con los resultados según la métrica para cada tipo de partición 
        """        

        train_metrics = []
        val_metrics = []
        test_metrics = []
        
        if kfold_indexes == None:
            X_train, y_train, X_val, y_val, X_test, y_test = self.train_val_test_split(X, y, random_state="hack")

            result = self.trainer(X_train=X_train, y_train=y_train, **params)
            train_r2 = self._r2(prediction=result, target=y_train)
            train_metrics.append((result, train_r2))
            
            resultat_val = self.validator(X_val=X_val, y_val=y_val) 
            val_r2 = self._r2(prediction=resultat_val, target=y_val)
            val_metrics.append((resultat_val, val_r2))
            
            resultat_test = self.validator(X_val=X_test, y_val=y_test)
            test_r2 = self._r2(prediction=resultat_test, target=y_test)
            test_metrics.append((resultat_test, test_r2))
                        
            return train_metrics, val_metrics, test_metrics
        
        
        for train_idx, test_idx in kfold_indexes:
            
            # trai part
            X_train = X[train_idx]
            y_train = y[train_idx]
            
            #val part
            X_val = X[test_idx]
            y_val = y[test_idx]
            
            result = self.trainer(X_train=X_train, y_train=y_train)
            train_r2 = self._r2(prediction=result, target=y_train)
            train_metrics.append((result, train_r2))
            
            resultat_val = self.validator(X_val=X_val, y_val=y_val) 
            val_r2 = self._r2(prediction=resultat_val, target=y_val)
            val_metrics.append((resultat_val, val_r2))
            
        return train_metrics, val_metrics, test_metrics

