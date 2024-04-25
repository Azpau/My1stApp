import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




#df=sns.load_dataset("iris")


rawData = pd.read_csv('FuelConsumption (1).csv')


rawData=rawData.drop(['Year'],axis=1)


rawData['ENGINE SIZE']=rawData['ENGINE SIZE'].astype('category')
rawData['CYLINDERS']=rawData['CYLINDERS'].astype('category')


# Import label encoder
from sklearn import preprocessing


# label_encoder object knows
# how to understand word labels.
label_encoder = preprocessing.LabelEncoder()


# Encode labels in column 'species'.
#rawData['Year']= label_encoder.fit_transform(rawData['Year'])
rawData['MAKE']= label_encoder.fit_transform(rawData['MAKE'])
rawData['MODEL']= label_encoder.fit_transform(rawData['MODEL'])
rawData['VEHICLE CLASS']= label_encoder.fit_transform(rawData['VEHICLE CLASS'])
rawData['TRANSMISSION']= label_encoder.fit_transform(rawData['TRANSMISSION'])
rawData['FUEL']= label_encoder.fit_transform(rawData['FUEL'])


from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
scalerSales = preprocessing.MinMaxScaler()      #use later for unscale data


rawData_sales = pd.DataFrame()
rawData_sales['FUEL CONSUMPTION'] = rawData['FUEL CONSUMPTION'].copy()          #use later for unscale data


scaled=scaler.fit_transform(rawData)
scaledSales=scalerSales.fit_transform(rawData_sales) #use later for unscale data
df_scaled = pd.DataFrame(scaled)
df_scaled.columns = rawData.columns
df_scaled.head()


from sklearn.model_selection import train_test_split


X=df_scaled.drop('FUEL CONSUMPTION', axis=1)
y=df_scaled['FUEL CONSUMPTION']


#training and testing split using all feature
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2) #stratify only for classification not regression


from sklearn.linear_model import LinearRegression
modellr = LinearRegression()
modellr.fit(X_train, y_train)
#y_pred = modellr.predict(X_test)


import pickle
pickle.dump(modellr,open("FuelConsumptionModel.h5","wb"))
print("Model Saved")
