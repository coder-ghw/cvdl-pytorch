#%%
import torch
# %%
# write data
import os
os.makedirs(os.path.join('..', 'data'), exist_ok=True)
file_name = os.path.join('../data', 'house_tiny.csv')
with open(file_name, 'w') as f:
    f.write('NumRooms,Alley,Price\n')
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

# %%
# read data
import pandas as pd
import numpy as np
data = pd.read_csv(file_name)
print(data)
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs.iloc[:, 0] = inputs.iloc[:, 0].fillna(inputs.iloc[:, 0].mean())
print(inputs)
inputs = pd.get_dummies(inputs, dummy_na=True).astype(np.float32)
print(inputs)

# %%
#转换为张量格式
import torch
X, y = torch.from_numpy(inputs.values), torch.from_numpy(outputs.values)
X, y

# %%

