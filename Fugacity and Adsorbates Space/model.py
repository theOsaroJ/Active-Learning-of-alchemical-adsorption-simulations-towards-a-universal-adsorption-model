import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# load model
model = load_model('gas_loading_prediction.h5')

# load training dataset
train_df = pd.read_csv("training.csv")
train_x = np.array(train_df.loc[:,"V2":"sig_eff"])
train_y = np.array(train_df.loc[:,"loading"])

# load test dataset
test_df = pd.read_csv("DL_AL.csv")
test_x = np.array(test_df.loc[:,"V2":"sig_eff"])
mof_name = test_df.loc[:,"MOF_name"]

# scale inputs/predictors
scaler = MinMaxScaler(feature_range=(0,1))
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

# gas loading prediction
pred_y = model.predict(test_x, verbose=0)
#print(mof_name)
#print('Values')
#print(pred_y)
with open("predictions.csv", "w") as pred:
	pred.write('MOF_name'+','+'prediciton'+'\n')
	for i in range(len(mof_name)):
		pred.write(str(mof_name[i])+","+str(pred_y[i])+'\n')
