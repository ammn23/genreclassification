'''
in your files, to work with dataset you can do:
  from dataset import load_data  
  X_train, X_test, y_train, y_test = load_data()

and then use X_train, X_test, y_train, y_test

'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def load_data():
  df = pd.read_csv("/Users/dilyaraarynova/MLProject/Dataset/features_3_sec.csv")
  df=df.drop(labels="filename",axis=1)
  
  # see first five values in the dataset:
  # # print(df.head())
  
  class_list=df.iloc[:,-1]
  converter=LabelEncoder()
  n=converter.fit_transform(class_list)
  print(n) # numeric vals of classes for genres column
  df.iloc[:, -1] = n # replace the genre column with numeric values
  
  # y = labels (genres)
  y = df['label'] 
  
  # X = all columns except 'label'
  X = df.loc[:, df.columns != 'label']
  
  # Save original column names
  cols = X.columns
  
  min_max_scaler = MinMaxScaler() # scales values between 0 and 1
  
  # fit_transform() returns a NumPy array (not a DataFrame)
  # # so, np_scaled contains the normalized feature values, but without column names or indexes.
  
  np_scaled = min_max_scaler.fit_transform(X)
  X = pd.DataFrame(np_scaled, columns=cols) # turns np_scaled back into a Pandas DataFrame.
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



# Shape of X_train: (6993, 58)
# Shape of X_test: (2997, 58)
# Shape of y_train: (6993,)
# Shape of y_test: (2997,)

# print(X_train.head())
# print(y_train.head())
