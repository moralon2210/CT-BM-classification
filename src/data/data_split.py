from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np

def split_train_val_test(df):
    x_train, X_temp, y_train, y_temp = train_test_split(
    df['ID'].tolist(), df['Label'].tolist(), test_size=0.3, stratify=df['Label'], random_state=42)

    x_val, x_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    return x_train, y_train, x_val, y_val, x_test, y_test







