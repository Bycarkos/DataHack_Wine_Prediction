import numpy as np 
import pandas as pd 
from .BaseModel import *
import xgboost as xgb
from typing import *
from utils import write_compare

class XGboost(BaseModel):
    
    def __init__(self, cfg:dict):
        super().__init__()
        
        # TODO DOCUMENTAR
        self._dict_results = []
        self._validations_results = []
        self._test_results = []
    
    
    def reset_params(self):
        self._dict_results = []
        self._validations_results = []
        self._test_results = []
        
    def custom_kfold_partition(self, df: pd.DataFrame, target_cols: list, n_splits: int = 5, shuffle: bool = True, random_state: Optional[Any] = None):
        return super().custom_kfold_partition(df, target_cols, n_splits, shuffle, random_state)
        
    def initialize_matrix(self,X:np.ndarray,y:np.ndarray):
        return xgb.DMatrix(X, label=y)
    
    
    def validator(self,model, X_val, y_val:Optional[np.ndarray]=None):
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
        
        if not y_val is None:rmse = self._rmse(y_val, y_pred)
        else:rmse = -1
        

        result = {
            'rmse': rmse,
            'gt': y_val,
            'pred': y_pred
            }

        
        return result, y_pred
            
    def trainer(self,X_train:np.ndarray, y_train:np.ndarray, **params):
        """
        Entrena un modelo de regresión XGBoost y devuelve el modelo entrenado, RMSE, número de muestras y número de características.

        Args:
            X_train (pd.DataFrame or np.array): El conjunto de entrenamiento de características.
            y_train (pd.Series or np.array): El conjunto de entrenamiento de etiquetas.
            params (dict): Los parámetros del modelo de regresión XGBoost.

        Returns:
            Un diccionario que contiene el modelo de regresión XGBoost entrenado, RMSE, número de muestras y número de características.
        """
        
        #dtrain = xgb.DMatrix(X_train, label=y_train)
        num_samples, num_features = X_train.shape

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_train, y_train)])

        y_pred = model.predict(X_train)
    

        rmse = self._rmse(y_train, y_pred)

        result = {
            'model': model,
            'rmse': rmse,
            'num_samples': num_samples,
            'num_features': num_features
            }

        return result, y_pred, model
    
    
    def train(self, X:np.ndarray, y:np.ndarray, write_compare_name:Optional[str]=None , kfold_indexes:Optional[None]=None, **params):
        
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
                            
        else:
            
        
            for fold_id, (train_idx, test_idx) in enumerate(kfold_indexes):
                
                # trai part
                X_train = X.iloc[train_idx]

                y_train = y.iloc[train_idx]
                
                #val part
                X_val = X.iloc[test_idx]
                y_val = y.iloc[test_idx]
                
                result, pred, net = self.trainer(X_train=X_train, y_train=y_train, **params)
                result["pred"] = pred
                result["gt"] = y_train
                result["fold_indexes"] = train_idx
                train_r2 = self._r2(prediction=pred, target=y_train)
                train_metrics.append((result, train_r2))

                resultat_val, y_pred = self.validator(model=net,X_val=X_val, y_val=y_val) 
                resultat_val["fold_indexes"] = test_idx
                val_r2 = self._r2(prediction=y_pred, target=y_val)
                val_metrics.append((resultat_val, val_r2))
                
                if write_compare_name is not None:
                    write_compare(X_val,y_pred, y_val, name_file=write_compare_name, fold=str(fold_id))
                
        self._dict_results = train_metrics
        self._validations_results = val_metrics
        self._test_results = test_metrics
            
