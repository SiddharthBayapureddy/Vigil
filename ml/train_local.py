# Trains a local instance of an XGBRegressor on the NASA CMAPSS dataset and uploads the model_weights, params, stats to S3

import xgboost as xgb
import pandas as pd
import numpy as np
import json
import os

import boto3

s3 = boto3.client("s3")

BUCKET     = "vigil-bucket"
TRAIN_KEY  = "raw-data/train/train.csv"
VAL_KEY    = "raw-data/train/validation.csv"
TRAIN_FILE = "train.csv"
VAL_FILE   = "validation.csv"
MODEL_FILE = "model.json"

MODEL_S3_KEY = "models/current/model.json"
STATS_S3_KEY = "processed/reference_stats.json"

DROP_COLS = ["engine_id", "cycle", "RUL"]
TARGET    = "RUL"


def download(s3 : boto3.client) -> None:
    print("Downloading datasets..")
    s3.download_file(BUCKET,TRAIN_KEY,TRAIN_FILE)
    s3.download_file(BUCKET,VAL_KEY,VAL_FILE)
    print(f"Downloaded the following datasets:\n{TRAIN_FILE},{VAL_FILE}")

    return None

def load():
    print("Loading and processing datasets")
    
    train_df = pd.read_csv(TRAIN_FILE)
    val_df   = pd.read_csv(VAL_FILE)

    X_train = train_df.drop(columns=DROP_COLS)
    y_train = train_df[TARGET]
    X_val   = val_df.drop(columns=DROP_COLS)
    y_val   = val_df[TARGET]

    print(f"      Train: {X_train.shape} | Val: {X_val.shape}")
    return X_train, y_train, X_val, y_val



# Training on XGBoost
def train(X_train : pd.DataFrame , y_train : pd.DataFrame , X_val : pd.DataFrame , y_val : pd.DataFrame) -> xgb.XGBRegressor:

    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.0,
        reg_lambda=1.0,
        random_state=67,
        tree_method="hist",
        n_jobs=-1,
        early_stopping_rounds = 20
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )
    return model

def evaluate(model : xgb.XGBRegressor, X_val : pd.DataFrame , y_val: pd.DataFrame) -> dict:
    
    print("Evaluating the model: ") 

    y_pred  = model.predict(X_val)
    mae     = float(np.mean(np.abs(y_val - y_pred)))
    mse     = float(np.mean((y_val - y_pred) ** 2))
    rmse    = float(np.sqrt(mse))
    ss_res  = float(np.sum((y_val - y_pred) ** 2))
    ss_tot  = float(np.sum((y_val - np.mean(y_val)) ** 2))
    r2      = float(1 - ss_res / ss_tot)

    print(f"      MAE  : {mae:.4f}")
    print(f"      RMSE : {rmse:.4f}")
    print(f"      R²   : {r2:.4f}")

    # Per-feature distribution stats 
    feature_stats = {}
    for feat in X_val.columns:
        vals = X_val[feat].dropna().values
        bins = np.linspace(vals.min(), vals.max(), 11)
        counts, _ = np.histogram(vals, bins=bins)
        feature_stats[feat] = {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals)),
            "min":  float(np.min(vals)),
            "max":  float(np.max(vals)),
            "percentiles": {
                str(p): float(np.percentile(vals, p))
                for p in [10, 25, 50, 75, 90, 95, 99]
            },
            "hist_bins":   bins.tolist(),
            "hist_counts": counts.tolist(),
        }

    # Prediction distribution — for MAE degradation check
    reference_stats = {
        "baseline_mae":  mae,
        "baseline_rmse": rmse,
        "baseline_r2":   r2,
        "n_val_samples": len(y_val),
        "feature_names": X_val.columns.tolist(),
        "features":      feature_stats,
        "prediction_distribution": {
            "mean": float(np.mean(y_pred)),
            "std":  float(np.std(y_pred)),
            "percentiles": {
                str(p): float(np.percentile(y_pred, p))
                for p in [10, 25, 50, 75, 90]
            },
        },
    }

    return reference_stats
    


def save_and_upload(s3 : boto3.client , model : xgb.XGBRegressor , stats : dict) -> None:

    print("Uploading model weights..")
    model.save_model(MODEL_FILE)
    with open("reference_stats.json" , "w") as file:
        json.dump(stats,file , indent=2)

    s3.upload_file(MODEL_FILE, BUCKET, MODEL_S3_KEY)
    s3.upload_file("reference_stats.json" , BUCKET , STATS_S3_KEY)

    print(f"Uploaded successfully")
    print(f"s3://{BUCKET}/{MODEL_S3_KEY}")
    print(f"s3://{BUCKET}/{STATS_S3_KEY}")

    # Deleting local files
    for file in [TRAIN_FILE,VAL_FILE,"reference_stats.json" , MODEL_FILE]:
        if os.path.exists(file):
            os.remove(file)

    return None


if __name__ == "__main__":

    download(s3)
    X_train, y_train, X_val, y_val = load()
    model  = train(X_train, y_train, X_val, y_val)
    stats  = evaluate(model, X_val, y_val)
    save_and_upload(s3, model, stats)

    print("\n\n")
    print("Done!")
    print("Reference_Stats for Drift: \n")
    print(f"Baseline MAE : {stats['baseline_mae']:.4f}")
    print(f"Baseline RMSE: {stats['baseline_rmse']:.4f}")
    print(f"Baseline R²  : {stats['baseline_r2']:.4f}")

