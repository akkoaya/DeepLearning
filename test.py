import os
import pandas as pd
data_file = os.path.join('data', 'house_tiny.csv')
data = pd.read_csv(data_file)
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean(numeric_only=True))
inputs = pd.get_dummies(inputs, dummy_na=True)
import torch

X = torch.tensor(inputs.to_numpy(dtype=float))
Y = torch.tensor(outputs.to_numpy(dtype=float))
print(X, Y)