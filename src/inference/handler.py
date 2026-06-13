# Parses an event of any engine, and engine cycle and returns the predicted rul and status

import io
import os
import json
import pickle
import sys

import boto3

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.append("/opt/python")
from features import build_features

BUCKET    = "vigil-bucket"
MODEL_KEY = "models/current/model.json"
SCALER_KEY = "processed/scaler.pkl"
MODEL_FILE = "/tmp/model.json"

s3 = boto3.client("s3")
cw = boto3.client("cloudwatch", region_name="us-east-1")

def load_model() -> xgb.XGBRegressor:

    s3.download_file(BUCKET,MODEL_KEY, MODEL_FILE)
    model = xgb.XGBRegressor()
    model.load_model(MODEL_FILE)
    return model


def load_scaler():
    buf = io.BytesIO()
    s3.download_fileobj(BUCKET, SCALER_KEY, buf)
    buf.seek(0)
    return pickle.load(buf)


def emit_metric(engine_id: int, predicted_rul: float) -> None:
    cw.put_metric_data(
        Namespace="Vigil",
        MetricData=[{
            "MetricName": "PredictedRUL",
            "Dimensions": [{"Name": "EngineId", "Value": str(engine_id)}],
            "Value": predicted_rul,
            "Unit": "Count"
        }]
    )


def lambda_handler(event, context):
    try:
        engine_id = event["engine_id"]
        cycle     = event["cycle"]
        sensors   = event["sensors"]

        row = {"engine_id": engine_id, "cycle": cycle, **sensors}
        df  = pd.DataFrame([row])

        model  = load_model()
        scaler = load_scaler()

        X, _, _ = build_features(df, scaler=scaler, fit=False)
        predicted_rul = float(model.predict(X)[0])
        predicted_rul = max(0.0, predicted_rul)

        emit_metric(engine_id, predicted_rul)

        return {
            "statusCode": 200,
            "body": json.dumps({
                "engine_id":     engine_id,
                "cycle":         cycle,
                "predicted_rul": round(predicted_rul, 2),
                "status":        "ok"
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e), "status": "error"})
        }