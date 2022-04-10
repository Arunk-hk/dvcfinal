#%%
import pandas as pd
import yaml
import os 
import io
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score,plot_confusion_matrix
from numpy import linalg as LA
from sklearn.preprocessing import MinMaxScaler
# Import label encoder
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import time
import pickle
#%%
params = yaml.safe_load(open("params.yaml"))['train']
print(params)
#%%
def generate_model_report(y_actual, y_predicted):
    print("Accuracy : " ,"{:.4f}".format(accuracy_score(y_actual, y_predicted)))     
    auc = roc_auc_score(y_actual, y_predicted)
    print("AUC : ", "{:.4f}".format(auc))
# %%
Train = pd.read_csv("data/Train.csv")
Test = pd.read_csv("data/Train.csv")
#%%
X_train = Train.iloc[:, :-1]
y_train = Train.iloc[:, -1]

X_test = Test.iloc[:, :-1]
y_test = Test.iloc[:, -1]
#%%
model = RandomForestClassifier(n_estimators=params['n_estimators'])
model.fit(X_train, y_train)
print("Training Report................ ")
generate_model_report(y_train,model.predict(X_train))
print("Testing Report................ ")
y_pred = model.predict(X_test)
generate_model_report(y_test,y_pred)
feat_importances_rf = pd.Series(model.feature_importances_, index=X_train.columns)
important_features_rf=feat_importances_rf.nlargest(50)
print('Feature Importance........') 
print(important_features_rf)  
#%%
with open('adult_model.pkl', "wb") as fd:
    pickle.dump(model, fd)

#%%    