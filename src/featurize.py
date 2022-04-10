#%%
import pandas as pd
import yaml
import os 
import io
#%%
params = yaml.safe_load(open("params.yaml"))['featurize']
print(params)
# %%
columns=params['featurelist']
# %%
Train = pd.read_csv("data/Train.csv")[columns]
Test = pd.read_csv("data/Train.csv")[columns]
# %%
Train.to_csv('data/Train.csv',index=False)  
Test.to_csv('data/Test.csv',index=False)  
# %%
