import torch
import torch.nn as nn
import torch.nn.functional as F
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt



class Classifier(nn.Module):
    def __init__(self):
        self.input_layer = nn.Conv2d(1,28,)
