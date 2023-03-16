import pandas as pd
import numpy as np
import math
from typing import *
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import mean_squared_error, r2_score
import os
import warnings
warnings.filterwarnings("ignore")

class Stats():
    
    def __init__(self, path:Optional[Path] = None, **params):
        if path is None:
            self._data = None
            
        else:
            self.load_file(path=path)
    
    def visualize_predictions(self,df_response: pd.DataFrame, metaclass:Optional[List] = None):
        dataframe = df_response
        if metaclass is None:
            sns.lineplot(data=dataframe[['REAL', 'RESPONSE']])
            plt.title("Comparación entre los valores reales y las predicciones")
            plt.xlabel("Índice")
            plt.ylabel("Valor")
            plt.legend(["Real", "Predicción"])
            plt.show()
        elif isinstance(metaclass, str):
            categories = dataframe[metaclass].unique()
            for category in categories:
                subset = dataframe[dataframe[metaclass] == category]
                sns.lineplot(data=subset[['REAL', 'RESPONSE']], label=category)
                plt.title(f"Comparación entre los valores reales y las predicciones para la categoría '{metaclass}'")
                plt.xlabel("Índice")
                plt.ylabel("Valor")
                plt.legend()
                plt.show()
        elif isinstance(metaclass, list):
            categories = dataframe.groupby(metaclass).groups.keys()
            for category in categories:
                subset = dataframe[dataframe[list(category)] == list(category)]
                sns.lineplot(data=subset[['REAL', 'RESPONSE']], label=", ".join(list(category)))
                plt.title(f"Comparación entre los valores reales y las predicciones para las categorías {', '.join(metaclass)}")
                plt.xlabel("Índice")
                plt.ylabel("Valor")
                plt.legend()
                plt.show()
    
    def write_verbose(self, df, metaclass, output_path="output.csv"):
        if isinstance(metaclass, str):
            groupby_columns = [metaclass]
        elif isinstance(metaclass, list):
            groupby_columns = metaclass
        else:
            groupby_columns = None

        if groupby_columns:
            groups = df.groupby(groupby_columns)
            results = pd.DataFrame(columns=["group"] + ["RMSE", "R2", "error_percent"])
            for name, group in groups:
                group_results = self.get_metrics(group)
                group_results.insert(0, name)
                results.loc[len(results)] = group_results
        else:
            results = pd.DataFrame(columns=["RMSE", "R2", "error_percent"])
            results.loc[0] = self.get_metrics(df)

        results.to_csv(output_path, index=False)

        
    def compare_metric_to_beat(self, df: pd.DataFrame, rmse_path: str = "metric_to_beat.csv"):
        # Calculamos el RMSE del dataframe que nos pasan
        rmse = ((df["REAL"] - df["RESPONSE"]) ** 2).mean() ** 0.5
        
        # Si no existe el fichero de RMSE lo creamos y guardamos el valor
        if not os.path.exists(rmse_path):
            self.write_metric_to_beat(rmse=rmse)
            print("Se ha creado el fichero de referencia de RMSE")
        
        # Si existe, leemos el valor y lo comparamos con el actual
        else:
            df = pd.read_csv(rmse_path)
            old_rmse = df["RMSE"].values[0]
            if rmse < old_rmse:
                self.write_metric_to_beat(rmse=rmse)
                print("Se ha actualizado el valor de RMSE de referencia")
            else:
                print("El RMSE del nuevo modelo no supera al de referencia")
   
    
    
    def write_response_file(self,x:pd.DataFrame, predictions:np.ndarray, name_file:Optional[Path]):
        # Agregar la columna "RESPONSE" al dataframe x
        print(x)
        print(predictions)
        x["RESPONSE"] = np.array(predictions)
        
        # Agrupar por las columnas especificadas y sumar las columnas "PRODUCCION" y "RESPONSE"
        group_cols = ["ID_FINCA", "VARIEDAD", "MODO", "TIPO", "COLOR", "SUPERFICIE"]
        sum_cols = ["PRODUCCION", "RESPONSE"]  if "PRODUCCION" in x.columns else ["RESPONSE"]
        grouped = x.groupby(group_cols, as_index=False)[sum_cols].sum()
        
        # Seleccionar las columnas específicas
        cols_to_keep = group_cols + sum_cols
        grouped = grouped[cols_to_keep]
        
        # Escribir el resultado en un archivo csv
        if name_file is None:
            name_file ="standard_validation_output.csv" 
        grouped.to_csv(name_file, index=False)
        
        # Calcular las métricas solicitadas
        resp_total = x["RESPONSE"].sum()
        if "PRODUCCION" in x.columns:
            prod_total = x["PRODUCCION"].sum()
            rmse = np.sqrt(mean_squared_error(x["PRODUCCION"], x["RESPONSE"]))
            r2 = r2_score(x["PRODUCCION"], x["RESPONSE"])
            perc_error = abs(prod_total - resp_total) / prod_total * 100
            
            # Escribir el mensaje verbose
            print(f"La suma de PRODUCCION es {prod_total} y la suma de RESPONSE es {resp_total}.")
            print(f"El RMSE es {rmse}, el R2 es {r2}, y el porcentaje de error de la diferencia de la suma de la producción total y la RESPONSE es {perc_error}%.")
        
        print(f"La suma de RESPONSE es {resp_total}.")



    def write_interval_confidence(self, df_response, metaclass):
        pass
    
    def write_model_decission_confidence(self, df_response, metaclasses):
        pass
    
    
    @staticmethod
    def get_metrics(df):
        y_true = df["REAL"].sum()
        y_pred = df["RESPONSE"].sum()

        RMSE = ((df["REAL"] - df["RESPONSE"]) ** 2).mean() ** 0.5
        R2 = r2_score(df["REAL"], df["RESPONSE"])
        error_percent = abs(y_true - y_pred) / y_true * 100

        return [RMSE, R2, error_percent]
             
    @staticmethod   
    def write_metric_to_beat(rmse):
        data = {"RMSE": [rmse]}
        df = pd.DataFrame(data)
        df.to_csv("metric_to_beat.csv", index=False)
        
        
    def load_file(self, path, **params) -> None:
        sep = params.get("sep", "|")
        self._data = pd.read_csv(path, sep=sep, names=["ID_FINCA", "VARIEDAD", "MODO", "COLOR","TIPO", "SUPERFICIE", "ID_ESTACION", "ALTITUD","ID_ZONA", "REAL", "RESPONSE"])
    
    
    
if __name__ == "__main__":
    stats = Stats("/home/carlos/PycharmProjects/DataHack_Wine_Prediction/outputs/export/kfold_0.csv")
    stats.compare_metric_to_beat(stats._data)
