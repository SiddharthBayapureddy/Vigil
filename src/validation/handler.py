# Trigged by sfn
# It downloads champion and challenger from s3, downloads val set
# Runs both models on val set, and checks each other score. Challenger must beat champion on both metrics to pass
# Returns {"promote" : True/False} back to sfns which decides next state

import json
import io
import pickle
import sys
import os

import boto3

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.append("/opt/python")
from features import build_features

BUCKET           = "vigil-bucket"
CHAMPION_KEY     = "models/current/model.json"
CHALLENGER_KEY   = "models/challenger/model.json"
CHAMPION_FILE    = "/tmp/champion.json"
CHALLENGER_FILE  = "/tmp/challenger.json"
SCALER_KEY       = "processed/scaler.pkl"
VAL_KEY          = "raw-data/train/validation.csv"
VAL_FILE         = "/tmp/validation.csv"

s3 = boto3.client("s3")


def load_model(s3_key : str , local_path : str) -> xgb.XGBRegressor:

    s3.download_file(BUCKET,s3_key,local_path)
    model = xgb.XGBRegressor()
    model.load_model(local_path)
    return model


def load_scaler():
    buf = io.BytesIO()
    s3.download_filejob(BUCKET,SCALER_KEY,buf)
    buf.seek(0)
    
    return pickle.load(buf)


def load_val_data(scaler):
    s3.download_file(BUCKET, VAL_KEY, VAL_FILE)
    val_df = pd.read_csv(VAL_FILE)
    X_val, y_val, _ = build_features(val_df, scaler=scaler, fit=False)
    return X_val, y_val


def nasa_score(y_true, y_pred) -> float:
    diff = y_pred - y_true
    score = np.where(diff < 0,
                     np.exp(-diff / 13) - 1,
                     np.exp(diff  / 10) - 1)
    return float(np.sum(score))


def evaluate_model(model: xgb.XGBRegressor,
                   X_val: pd.DataFrame,
                   y_val: pd.Series) -> dict:
    
    y_pred = model.predict(X_val)
    mae    = float(np.mean(np.abs(y_val - y_pred)))
    nasa   = nasa_score(y_val.values, y_pred)
    return {"mae": round(mae, 4), "nasa": round(nasa, 4)}



def lambda_handler(event, context):
    try:
        scaler     = load_scaler()
        X_val, y_val = load_val_data(scaler)

        champion   = load_model(CHAMPION_KEY,   CHAMPION_FILE)
        challenger = load_model(CHALLENGER_KEY, CHALLENGER_FILE)

        champ_metrics = evaluate_model(champion,   X_val, y_val)
        chal_metrics  = evaluate_model(challenger, X_val, y_val)

        # Challenger must beat champion on both metrics
        promote = (
            chal_metrics["mae"]   < champ_metrics["mae"] and
            chal_metrics["nasa"]  < champ_metrics["nasa"]
        )

        print(f"Champion  — MAE: {champ_metrics['mae']} | NASA: {champ_metrics['nasa']}")
        print(f"Challenger — MAE: {chal_metrics['mae']} | NASA: {chal_metrics['nasa']}")
        print(f"Promote: {promote}")

        return {
            "promote":          promote,
            "champion_mae":     champ_metrics["mae"],
            "champion_nasa":    champ_metrics["nasa"],
            "challenger_mae":   chal_metrics["mae"],
            "challenger_nasa":  chal_metrics["nasa"],
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e), "status": "error"})
        }