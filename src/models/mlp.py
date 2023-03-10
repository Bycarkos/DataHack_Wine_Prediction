import torch.nn as nn 
import torch
from torchmetrics import R2Score, 
import pandas as pd 
import numpy as np 
from typing import *
from pathlib import Path
from utils import write_compare_file
from overrides import overrides


class MLP(nn.Module):
    
    def __init__(self, input_size:int, hidden_size:int, output_size:int, nlayers:int) -> None:
        super().__init__()
        
        #Metrics
        self._rmse = lambda y_hat, y: torch.sqrt(torch.mean((y_hat - y)**2))
        self._r2 = R2Score() #preds, target
        
        #
        self._nlayers = nlayers
        