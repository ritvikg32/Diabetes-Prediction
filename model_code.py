#importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib


df=pd.read_csv(r'C:\Users\ritvi\Documents\Django model\Diabetes\diabetes.csv')

#Replacing 0 by Nan
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']]=df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

#Filling Nan Values by Performing Data Imputation

df['Glucose'].fillna(df['Glucose'].median(), inplace = True)
df['BloodPressure'].fillna(df['BloodPressure'].median(), inplace = True)
df['SkinThickness'].fillna(df['SkinThickness'].median(), inplace = True)
df['Insulin'].fillna(df['Insulin'].median(), inplace = True)
df['BMI'].fillna(df['BMI'].mean(), inplace = True)


#Removing Outliers

Q1 = np.percentile(df['Pregnancies'], 25, 
                   interpolation = 'midpoint') 
  
Q3 = np.percentile(df['Pregnancies'], 75,
                   interpolation = 'midpoint') 
IQR = Q3 - Q1 

# Above Upper bound
upper = df['Pregnancies'] >= (Q3+1.5*IQR)
print(np.where(upper))

df["Pregnancies"] = np.where(df["Pregnancies"]>Q3, df['Pregnancies'].mean(),df['Pregnancies'])

# Defining Features of Data
X=df.drop('Outcome',axis=1)

# Defining Target Variable of Data
Y=df['Outcome']

# Importing Libraries for train test split
from sklearn.model_selection import train_test_split

# Train data set 80% and test data set 20%
X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.20,random_state=101)


#Feature Scaling 
#Standard Scalar 

from sklearn.preprocessing import StandardScaler
std=StandardScaler()

X_train=std.fit_transform(X_train)
X_test=std.transform(X_test)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()

final_model = lr.fit(X_train,Y_train) # Model is being made wrt the values of training data


# Y_pred=lr.predict(X_test) # Predicting values of Y


# Saving the model locally 
filename = 'diabetes_model.sav'
joblib.dump(lr, filename)

#Saving standard scalar 
std_file = 'standard_scaler.sav'
joblib.dump(std,std_file)
