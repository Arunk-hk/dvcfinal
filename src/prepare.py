#%%
from matplotlib.pyplot import axis
import pandas as pd
import yaml
import os 
import io
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

#%%
params = yaml.safe_load(open("params.yaml"))['prepare']
print(params)
data = pd.read_csv("data/adult.csv")
#%%
data_org=data.copy()

numerical_list=[]
categorical_list=[]
for i in data.columns.tolist():
    if data[i].dtype=='object':
        categorical_list.append(i)
    else:
        numerical_list.append(i)
print('Number of categorical features:', str(len(categorical_list)))
print('Number of numerical features:', str(len(numerical_list)))

label_encoder = preprocessing.LabelEncoder()
for col in categorical_list:
    data[col]=label_encoder.fit_transform(data[col])
# %%
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = params['split'], random_state = 12)  
X_train=pd.concat([X_train,y_train],axis=1)
X_train=pd.concat([X_test,y_test],axis=1)
X_train.to_csv('data/Train.csv',index=False)  
X_test.to_csv('data/Test.csv',index=False)  
# %%
