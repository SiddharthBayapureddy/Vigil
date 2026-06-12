# Trains a local instance of an XGBRegressor on the NASA CMAPSS dataset and uploads the model_weights, params, stats to S3

import xgboost as xgb
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timezone

import boto3

from features import build_features

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

def load(s3 : boto3.client):
    print("Loading and processing datasets")
    
    train_df = pd.read_csv(TRAIN_FILE)
    train_df["RUL"] = train_df["RUL"].clip(upper=125)

    val_df   = pd.read_csv(VAL_FILE)
    val_df["RUL"] = val_df["RUL"].clip(upper=125)

    X_train, y_train, scaler = build_features(train_df, fit=True, s3=s3)
    X_val,   y_val,   _      = build_features(val_df,   scaler=scaler)

    print(f"      Train: {X_train.shape} | Val: {X_val.shape}")
    return X_train, y_train, X_val, y_val


def nasa_custom_objective(y_true, y_pred):
    """
    Custom asymmetric objective function matching the NASA evaluation metric.
    Heavily penalizes over-estimation (predicting an engine is safe when it is about to fail).
    Returns gradient and hessian for XGBoost.
    """
    diff = y_pred - y_true
    
    # Gradients (first derivative)
    grad = np.where(diff < 0, 
                    (-1.0 / 13.0) * np.exp(-diff / 13.0), 
                    (1.0 / 10.0) * np.exp(diff / 10.0))
    
    # Hessians (second derivative)
    hess = np.where(diff < 0, 
                    (1.0 / 169.0) * np.exp(-diff / 13.0), 
                    (1.0 / 100.0) * np.exp(diff / 10.0))
    
    return grad, hess


# Training on XGBoost
def train(X_train : pd.DataFrame , y_train : pd.DataFrame , X_val : pd.DataFrame , y_val : pd.DataFrame) -> xgb.XGBRegressor:

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.0,
        reg_lambda=1.0,
        random_state=67,
        tree_method="hist",
        n_jobs=-1,
        early_stopping_rounds=20,
        objective=nasa_custom_objective # Applied the asymmetric objective
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )
    return model


def nasa_score(y_true, y_pred):
    diff = y_pred - y_true
    score = np.where(diff < 0,
                     np.exp(-diff / 13) - 1,
                     np.exp(diff  / 10) - 1)
    return float(np.sum(score))

def evaluate(model : xgb.XGBRegressor, X_val : pd.DataFrame , y_val: pd.DataFrame) -> dict:
    
    print("Evaluating the model: ") 

    y_pred  = model.predict(X_val)
    mae     = float(np.mean(np.abs(y_val - y_pred)))
    mse     = float(np.mean((y_val - y_pred) ** 2))
    rmse    = float(np.sqrt(mse))
    ss_res  = float(np.sum((y_val - y_pred) ** 2))
    ss_tot  = float(np.sum((y_val - np.mean(y_val)) ** 2))
    r2      = float(1 - ss_res / ss_tot)
    nasa = nasa_score(y_val,y_pred)

    print(f"      MAE  : {mae:.4f}")
    print(f"      RMSE : {rmse:.4f}")
    print(f"      R²   : {r2:.4f}")
    print(f"      NASA : {nasa:.4f}")

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

    importance = dict(zip(X_val.columns, model.feature_importances_))
    top_10 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

    # Prediction distribution — for MAE degradation check
    reference_stats = {
        "baseline_mae":  mae,
        "baseline_rmse": rmse,
        "baseline_r2":   r2,
        "baseline_nasa": nasa,
        "n_val_samples": len(y_val),
        "feature_names": X_val.columns.tolist(),
        "features":      feature_stats,
        "top_features":  {k: round(float(v), 6) for k, v in top_10},
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

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    try:
        s3.copy_object(
            Bucket=BUCKET,
            CopySource={"Bucket": BUCKET, "Key": MODEL_S3_KEY},
            Key=f"models/archive/model_{timestamp}.json"
        )
        print(f"      Archived → s3://{BUCKET}/models/archive/model_{timestamp}.json")
    except Exception:
        pass  # First run, nothing to archive yet

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
    X_train, y_train, X_val, y_val = load(s3)
    model  = train(X_train, y_train, X_val, y_val)
    stats  = evaluate(model, X_val, y_val)
    save_and_upload(s3, model, stats)

    print("\n\n")
    print("Done!")
    print("Reference_Stats for Drift: \n")
    print(f"Baseline MAE : {stats['baseline_mae']:.4f}")
    print(f"Baseline RMSE: {stats['baseline_rmse']:.4f}")
    print(f"Baseline R²  : {stats['baseline_r2']:.4f}")
    print(f"NASA Score   : {stats['baseline_nasa']:.4f}")