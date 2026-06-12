# Adding temporal context by computing rolling statistics per each engine:
# - rolling mean : smooths noise, captures trend direction
# - rolling std : captures volatility (engines degrade erratically near failure)
# - rolling min/max : captures extremes within a window


# 2 windows: 5 cycles & 10 cycles 
# Total 14 sensors x 4 stats x 2 windows = 112 new features on top of original 24 cols

import io
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import boto3

BUCKET = "vigil-bucket"
SCALER_KEY = "processed/scaler.pkl"

SENSOR_COLS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8",
    "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
    "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]

WINDOWS = [5,10]
DROP_COLS = ["engine_id" , "cycle" , "RUL"]
TARGET = "RUL"

# Feature Engineering

def add_rolling_features(df : pd.DataFrame) -> pd.DataFrame:
    ''' Adds rolling mean,std,min,max per sensor per engine
    Groups engine_id so windows don't across other engines'''

    df = df.copy()
    new_cols = {}

    # Delta from each engine's own cycle-1 baseline
    for sensor in SENSOR_COLS:
        baseline = df.groupby("engine_id")[sensor].transform("first")
        new_cols[f"{sensor}_delta_from_start"] = df[sensor] - baseline

    # Normalized cycle position (0.0 → 1.0) per engine
    # Replaced hardcoded MAX_CYCLE with raw cycle. The StandardScaler will handle dynamic normalization.
    new_cols["cycle_norm"] = df["cycle"]

    for window in WINDOWS:
        for sensor in SENSOR_COLS:

            # Grouping by engine_id to ensure engine_i doesn't interfere with engine_i+1
            # min_periods : If no enough data, takes the stat for the data of min_periods
            # If no min_periods, default rolling behaviour causes the first window elements to be NaN, with min_period = 1, maintains the sats with no NaNs

            grouped = df.groupby("engine_id")[sensor]
            new_cols[f"{sensor}_mean_{window}"] = grouped.transform(lambda x: x.rolling(window, min_periods=1).mean())
            new_cols[f"{sensor}_std_{window}"]  = grouped.transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))  # std needs a min of two values
            new_cols[f"{sensor}_min_{window}"]  = grouped.transform(lambda x: x.rolling(window, min_periods=1).min())
            new_cols[f"{sensor}_max_{window}"]  = grouped.transform(lambda x: x.rolling(window, min_periods=1).max())

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)



def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Returns all feature column names (original + rolling), removing the dropped columns"""
    return [c for c in df.columns if c not in DROP_COLS]



# Standard Scaler
# X_scaled = (X - mean) / std

# Training: fit scaler on X_train and save it to S3
# Inference: load scalar from S3 and transform incoming X

def fit_scaler(X: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler

def apply_scaler(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    return pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)

def save_scaler(scaler: StandardScaler, s3: boto3.client) -> None:
    buf = io.BytesIO()
    pickle.dump(scaler, buf)
    buf.seek(0)
    s3.upload_fileobj(buf, BUCKET, SCALER_KEY)
    print(f"      Scaler saved → s3://{BUCKET}/{SCALER_KEY}")


def load_scaler(s3: boto3.client) -> StandardScaler:
    buf = io.BytesIO()
    s3.download_fileobj(BUCKET, SCALER_KEY, buf)
    buf.seek(0)
    return pickle.load(buf)


# Pipeline

def build_features(df: pd.DataFrame , scaler : StandardScaler = None,
                   fit: bool = False , s3 : boto3.client = None) -> tuple[pd.DataFrame , pd.Series , StandardScaler]:
    
    '''Entire rolling features pipeline. Fit = false during inference, Fit = true during training
    Returns X - Feature df , y - RUL series , scaler - fitted scaler'''
    

    df = add_rolling_features(df)
    feature_cols = get_feature_columns(df)

    X = df[feature_cols]
    
    if TARGET in df.columns:
        y = df[TARGET]
    else:
        y = None

    if fit:
        scaler = fit_scaler(X)
        if s3:
            save_scaler(scaler,s3)
        
    if scaler is not None:
        X = apply_scaler(X , scaler)

    return X,y,scaler