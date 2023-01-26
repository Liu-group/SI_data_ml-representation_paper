# Example script how to load a trained model contained in this SI.zip and predict utilizing the input features.
# This example used the SS descriptor model to correct the b3lyp calculated absorption energies.
# The trained ML model requires scaled input features, below is shown how to use the input features to define the scaler.


import pickle
#pickle version 4.0
import sklearn
#sklearn version 1.0
import numpy as np
import pandas as pd

#load model pkl
model_pkl = open('./ROAS_models/ROAS_b3lyp_SS_RF.pkl', 'rb')
model = pickle.load(model_pkl)

#load feature csv
features=np.genfromtxt('./ROAS_feature/ROAS_b3lyp_SS.csv', delimiter=',')

#load delta cvs
delta=pd.read_csv('./ROAS.csv')['b3lyp delta[eV]']
#scale input features
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(features)
features_scaled= scaler.transform(features)
#split into train, test
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(features_scaled, delta, test_size=0.20, random_state=42)
#predict for training set
pred_train=model.predict(X_train)
#predict for test set
pred_train=model.predict(X_test)