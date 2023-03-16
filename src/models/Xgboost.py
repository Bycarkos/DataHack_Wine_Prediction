from pathlib import Path

import pandas as pd
from overrides import overrides
import os

from .BaseModel import *
import xgboost as xgb
from typing import *

from utils import *

class XGboost(BaseModel):
    
    def __init__(self, cfg: dict, export_dir: Path):
        super().__init__()

        # TODO DOCUMENTAR
        self.cfg = cfg

        self._dict_results = []
        self._validations_results = []
        self._test_results = []
        

        self.export_dir = export_dir
    
    def reset_params(self):
        self._dict_results = []
        self._validations_results = []
        self._test_results = []
        
    def custom_kfold_partition(self, df: pd.DataFrame, target_cols: Union[str, list], n_splits: int = 5, shuffle: bool = True, random_state: Optional[Any] = None):
        return super().custom_kfold_partition(df, target_cols, n_splits, shuffle, random_state)

    @staticmethod
    def initialize_matrix(x: np.ndarray, y: np.ndarray):
        return xgb.DMatrix(x, label=y)

    @overrides(check_signature=False)
    def trainer(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray):
        """
        Entrena un modelo de regresión XGBoost y devuelve el modelo entrenado, RMSE, número de muestras y número de características.

        Args:
            x_train (pd.DataFrame or np.array): El conjunto de entrenamiento de características.
            y_train (pd.Series or np.array): El conjunto de entrenamiento de etiquetas.
            params (dict): Los parámetros del modelo de regresión XGBoost.

        Returns:
            Un diccionario que contiene el modelo de regresión XGBoost entrenado, RMSE, número de muestras y número de características.
        """
        
        #dtrain = xgb.DMatrix(X_train, label=y_train)
        num_samples, num_features = x_train.shape

        model = xgb.XGBRegressor(**self.cfg)
        model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)], early_stopping_rounds=10)  # Evaluation on train set, validation done apart

        y_pred = model.predict(x_train)

        rmse = self._rmse(y_train, y_pred)

        result = {
            'model': model,
            'rmse': rmse,
            'num_samples': num_samples,
            'num_features': num_features
            }

        return result, y_pred, model

    @overrides(check_signature=False)
    def validator(self, model, x_test, y_test: Optional[np.ndarray] = None, write_compare:Optional[bool] = True):
        """
        Valida un modelo de regresión XGBoost y devuelve la métrica de validación RMSE.

        Args:
            model (xgb.XGBRegressor): El modelo de regresión XGBoost entrenado.
            x_val (pd.DataFrame or np.array): El conjunto de validación de características.
            y_val (pd.Series or np.array): El conjunto de validación de etiquetas.

        Returns:
            La métrica de validación (RMSE).
        """


        print("--> Test")
        y_pred = model.predict(x_test)
        
        if write_compare:
            write_compare_fold_dir = self.export_dir
            if not os.path.exists(write_compare_fold_dir):
                os.mkdir(write_compare_fold_dir)

            print("Write the comparison for RSME...")
            write_compare_name_test = str(write_compare_fold_dir / "standard_test.csv")
            write_response_file(x=x_test, predictions=y_pred, name_file=write_compare_name_test)
            

        if y_test is not None:
            rmse = self._rmse(y_test, y_pred)
        else:
            rmse = -1

        result = {
            'rmse': rmse,
            'gt': y_test,
            'pred': y_pred
        }

        return result, y_pred

    def train(self, x_train:pd.DataFrame, y_train:pd.DataFrame, x_val: Optional[pd.DataFrame], y_val: Optional[np.ndarray], write_compare: bool = False, kfold_indexes: Optional[list] = None):
        """
        Entrena un modelo de regresión XGBoost y devuelve tres listas con la prediccion según la métrica RMSE.

        Args:
            x (pd.DataFrame or np.array): El conjunto de datos entero
            y (pd.Series or np.array): El conjunto de etiquetas.
            model_config (dict): Los parámetros del modelo de regresión XGBoost.
            write_compare:
            kfold_indexes:

        Returns:
            Tres listas con los resultados según la métrica para cada tipo de partición 
        """         

        train_metrics = []
        val_metrics = []
        
        if not kfold_indexes:
            print("--> Training")
            result, y_pred, model = self.trainer(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
            train_r2 = self._r2(prediction=y_pred, target=y_train)
            train_metrics.append((result, train_r2))

            print("--> Validation")
            resultat_val, y_pred = self.validator(model=model, x_test=x_val, y_test=y_val)
            val_r2 = self._r2(prediction=y_pred, target=y_val)
            val_metrics.append((resultat_val, val_r2))

            if write_compare:
                write_compare_fold_dir = self.export_dir
                if not os.path.exists(write_compare_fold_dir):
                    os.mkdir(write_compare_fold_dir)

                print("Write the comparison for RSME...")
                write_compare_name_val = str(write_compare_fold_dir / "standard_val.csv")
                write_response_file(x = x_val, predictions=y_pred, name_file=write_compare_name_val)

        
        else:
            print("Kfold indexes, iterating over splits...")
            for fold_id, (train_idx, test_idx) in enumerate(kfold_indexes):
                print("--> Fold_id:", fold_id)
                
                # Train split
                x_train = x_train.iloc[train_idx]
                y_train = y_train.iloc[train_idx]
                
                # Valid split
                x_val = x_train.iloc[test_idx]
                y_val = y_train.iloc[test_idx]

                print("--> Training")
                result, y_pred, model = self.trainer(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
                train_r2 = self._r2(prediction=y_pred, target=y_train)
                train_metrics.append((result, train_r2))

                print("--> Validation")
                resultat_val, y_pred = self.validator(model=model, x_val=x_val, y_val=y_val)
                val_r2 = self._r2(prediction=y_pred, target=y_val)
                val_metrics.append((resultat_val, val_r2))
                print(f"In the Validation r2={val_r2}")

                if write_compare:
                    write_compare_fold_dir = self.export_dir
                    if not os.path.exists(write_compare_fold_dir):
                        os.mkdir(write_compare_fold_dir)

                    write_compare_name = str(write_compare_fold_dir / f"Kfold_{fold_id}.csv")
                    
                    print(f"FOLD_{fold_id}: Write the comparison for RSME...")
                    write_response_file(x = x_val, predictions=y_pred, name_file=write_compare_name)
                
            best_model_idx = get_best_model(val_metrics)
            model = model._dict_results[best_model_idx][0]["model"]    
            
        self._dict_results = train_metrics
        self._validations_results = val_metrics
        
        return model    
    
