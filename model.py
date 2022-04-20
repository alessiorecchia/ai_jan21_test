import torch
import torch.nn as nn
import torch.nn.functional as F
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt



class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.input_layer = nn.Conv2d(1,28,kernel_size=7)
        self.conv1 = nn.Conv2d(28, 56, 7)
        self.conv2 = nn.Conv2d(56,112,7)
        self.drop = nn.Dropout2d(0.1)
        self.layer1 = nn.Linear(111,60)
        self.layer2 = nn.Linear(60,10)

    def forward(self, x):
        x = F.relu(self.drop(self.input_layer(x)))
        x

