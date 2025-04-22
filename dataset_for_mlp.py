import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def load_data():
    # Load the 30 sec and 3 sec CSV files
    df_30_sec = pd.read_csv(
        "C:\\Users\\amina\\Datasets\\features_30_sec.csv")
    df_3_sec = pd.read_csv(
        "C:\\Users\\amina\\Datasets\\features_3_sec.csv")

    # Print column names to debug
    print("Columns in 30 sec dataset:", df_30_sec.columns.tolist())
    print("Columns in 3 sec dataset:", df_3_sec.columns.tolist())

    # Drop the filename columns (if exists)
    if 'filename' in df_30_sec.columns:
        df_30_sec = df_30_sec.drop(columns=['filename'])
    else:
        first_col_30 = df_30_sec.columns[0]
        df_30_sec = df_30_sec.drop(columns=[first_col_30])

    if 'filename' in df_3_sec.columns:
        df_3_sec = df_3_sec.drop(columns=['filename'])
    else:
        first_col_3 = df_3_sec.columns[0]
        df_3_sec = df_3_sec.drop(columns=[first_col_3])

    # The last column in both datasets should be the genre label
    last_col_30_sec = df_30_sec.columns[-1]
    last_col_3_sec = df_3_sec.columns[-1]

    print(f"Last column in 30 sec dataset (expected to be genre): {last_col_30_sec}")
    print(f"Last column in 3 sec dataset (expected to be genre): {last_col_3_sec}")

    # Convert genres to numeric labels for both datasets
    converter = LabelEncoder()
    y_30_sec = converter.fit_transform(df_30_sec[last_col_30_sec])
    y_3_sec = converter.fit_transform(df_3_sec[last_col_3_sec])

    print("Genre labels from 30 sec dataset converted to:", y_30_sec[:10], "...")
    print("Genre labels from 3 sec dataset converted to:", y_3_sec[:10], "...")

    # Drop the genre column from both datasets
    df_30_sec = df_30_sec.drop(columns=[last_col_30_sec])
    df_3_sec = df_3_sec.drop(columns=[last_col_3_sec])

    # Split the data separately for both datasets
    X_30_sec_train, X_30_sec_test, y_30_sec_train, y_30_sec_test = train_test_split(df_30_sec, y_30_sec, test_size=0.3,
                                                                                    random_state=42)
    X_3_sec_train, X_3_sec_test, y_3_sec_train, y_3_sec_test = train_test_split(df_3_sec, y_3_sec, test_size=0.3,
                                                                                random_state=42)

    # Combine the features and labels from both datasets
    X_combined = pd.concat([X_30_sec_train, X_3_sec_train], axis=0, ignore_index=True)
    y_combined = np.concatenate([y_30_sec_train, y_3_sec_train])

    X_test_combined = pd.concat([X_30_sec_test, X_3_sec_test], axis=0, ignore_index=True)
    y_test_combined = np.concatenate([y_30_sec_test, y_3_sec_test])

    # Convert X_combined to numeric values (handle any non-numeric columns)
    for col in X_combined.columns:
        try:
            X_combined[col] = pd.to_numeric(X_combined[col])
        except ValueError:
            print(f"Warning: Column '{col}' contains non-numeric values. Dropping this column.")
            X_combined = X_combined.drop(columns=[col])

    print(f"Feature matrix shape after cleaning: {X_combined.shape}")

    # Scale the features
    min_max_scaler = MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(X_combined)
    X_combined = pd.DataFrame(np_scaled, columns=X_combined.columns)

    # Scale the test data in the same way
    X_test_scaled = min_max_scaler.transform(X_test_combined)
    X_test_combined = pd.DataFrame(X_test_scaled, columns=X_test_combined.columns)

    return X_combined, X_test_combined, y_combined, y_test_combined

load_data()