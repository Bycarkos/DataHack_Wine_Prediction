import torch.nn as nn 
import torch
from torchmetrics import R2Score
import pandas as pd 
import numpy as np 
from typing import *
from pathlib import Path
from utils import write_compare_file
from overrides import overrides
from hydra.utils import instantiate


class Forecast_Embedding(nn.Module):
    
    def __init__(self, cfg) -> None:
        super().__init__()
        
        #Metrics
        self._mse = lambda y_hat, y: (torch.mean((y_hat - y)**2))
        self._r2 = R2Score() #preds, target
        
        # Model Params
        self._nlayers = cfg.nlayers
        self._input_size = cfg.input_size
        self._hidden = cfg.hidden_size
        self._out_size = cfg.out_size
        self._forecast_window = cfg.window
        
        # Linears
        self._lin = nn.Linear(in_features=self._input_size*self._forecast_window, out_features=self._out_size, bias=True)
        self._flin_out = nn.Linear(in_features=self._out_size, out_features=self._out_size)
        
        # non_linears
        self._act = instantiate(cfg.activation_layer)
        self._layer_norm = instantiate(cfg.layer_norm)
        
        
        