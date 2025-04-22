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
    # Load the CSV file with header
    df = pd.read_csv("C:\\Users\\User\\Desktop\\Spring 2025\\ml\\genreclassification\\dataset\\features_3_sec.csv")
    
    # Print column names to debug
    print("Column names:", df.columns.tolist())
    
    # Assuming the first column is filename and the last column is the genre
    # Extract features and labels
    
    # Drop the filename column
    if 'filename' in df.columns:
        df = df.drop(columns=['filename'])
    else:
        # If 'filename' isn't the exact column name, drop the first column
        first_col = df.columns[0]
        df = df.drop(columns=[first_col])
    
    # The last column should be the genre label
    # Check the name of the last column
    last_col = df.columns[-1]
    print(f"Last column (expected to be genre): {last_col}")
    
    # Convert genres to numeric labels
    converter = LabelEncoder()
    y = converter.fit_transform(df[last_col])
    print("Genre labels converted to:", y[:10], "...")  # Print first 10 numeric values
    
    # Get feature columns (all except the genre column)
    X = df.drop(columns=[last_col])
    
    # Convert X to numeric values (handle any non-numeric columns)
    # This is a safer approach to handle potential string values
    for col in X.columns:
        try:
            X[col] = pd.to_numeric(X[col])
        except ValueError:
            print(f"Warning: Column '{col}' contains non-numeric values. Dropping this column.")
            X = X.drop(columns=[col])
    
    print(f"Feature matrix shape after cleaning: {X.shape}")
    
    # Scale the features
    min_max_scaler = MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(X)
    X = pd.DataFrame(np_scaled, columns=X.columns)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Ensure y_train and y_test are proper integer types
    y_train = pd.Series(y_train).astype(int)
    y_test = pd.Series(y_test).astype(int)
    
    return X_train, X_test, y_train, y_test


# Shape of X_train: (6993, 58)
# Shape of X_test: (2997, 58)
# Shape of y_train: (6993,)
# Shape of y_test: (2997,)

# print(X_train.head())
# print(y_train.head())
