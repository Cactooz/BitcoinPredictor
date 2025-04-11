import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length])
    
    return np.array(X_seq), np.array(y_seq)
    
def create_data_loader(feature_columns, target_column, batch_size=64, lookback_length=365, test_length=365):
    # Load and sort the data
    data = pd.read_csv("./data/BTC Data 2013-12-27 2025-04-01.csv")
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date')

    print(f"Data range: {data['date'].min()} to {data['date'].max()}")
    print(f"Total samples: {len(data)}")

    data['dayOfWeek'] = data['date'].dt.dayofweek
    data['dayOfMonth'] = data['date'].dt.day
    data['month'] = data['date'].dt.month
    data['quarter'] = data['date'].dt.quarter
    data['isWeekend'] = data['dayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

    X = data[feature_columns].values
    y = data[target_column].values.reshape(-1, 1)

    X_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    X_normalized = X_scaler.fit_transform(X)
    y_normalized = y_scaler.fit_transform(y)

    X_seq, y_seq = create_sequences(X_normalized, y_normalized, lookback_length)

    test_dates = data['date'].iloc[-test_length:]

    train_dataset = TimeSeriesDataset(X_seq[:-test_length], y_seq[:-test_length])
    test_dataset = TimeSeriesDataset(X_seq[-test_length:], y_seq[-test_length:])

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader, test_dates, X_scaler, y_scaler