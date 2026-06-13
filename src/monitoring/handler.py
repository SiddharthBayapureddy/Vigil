# Triggered by Eventbridge cron
# Downloads the necessary model files, and runs drift checks, and writes a drift report.
# If drift is detected, invokes another lambda function for that purpose

import json
import sys
import os
import boto3
import io
import pickle

import pandas as pd
import xgboost as xgb
from datetime import datetime

sys.path.append("/opt/python")
from features import build_features
from drift import run_drift_checks

BUCKET           = "vigil-bucket"
INCOMING_KEY     = "raw-data/incoming/test.csv"
INCOMING_FILE    = "/tmp/test.csv"
MODEL_KEY        = "models/current/model.json"
MODEL_FILE       = "/tmp/model.json"
SCALER_KEY       = "processed/scaler.pkl"
STATS_KEY        = "processed/reference_stats.json"
STATS_FILE       = "/tmp/reference_stats.json"
NARRATION_LAMBDA = "vigil-narration"
STEPFUNCTIONS_ARN = "STEP_FUNCTIONS_ARN_PLACEHOLDER"

s3 = boto3.client("s3")
lam = boto3.client("lambda" , region_name="us-east-1")
sfn = boto3.client("stepfunctions", region_name="us-east-1")


def load_artifacts():

    s3.download_file(BUCKET, INCOMING_KEY, INCOMING_FILE)
    s3.download_file(BUCKET, MODEL_KEY, MODEL_FILE)
    s3.download_file(BUCKET, STATS_KEY, STATS_FILE)

    incoming_df = pd.read_csv(INCOMING_FILE)

    ref_stats = {}
    with open(STATS_FILE , "r") as file:
        ref_stats = json.load(file)

    model = xgb.XGBRegressor()
    model.load_model(MODEL_FILE)

    buf = io.BytesIO()
    s3.download_fileobj(BUCKET, SCALER_KEY, buf)
    buf.seek(0)
    scaler = pickle.load(buf)

    return incoming_df,ref_stats,scaler,model


def save_drift_report(report: dict) -> str:
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    key = f"processed/drift_reports/report_{timestamp}.json"
    body = json.dumps(report, indent=2)
    s3.put_object(Bucket=BUCKET, Key=key, Body=body)
    return key


def invoke_narration(report: dict):
        
    lam.invoke(
        FunctionName=NARRATION_LAMBDA,
        InvocationType="Event",  # async
        Payload=json.dumps(report)
    )


def start_retraining(report: dict) -> None:
    sfn.start_execution(
        stateMachineArn=STEPFUNCTIONS_ARN,
        input=json.dumps({"drift_report": report})
    )


def lambda_handler(event, context):
    try:
        incoming_df, ref_stats,scaler, model = load_artifacts()

        incoming_engineered, _, _ = build_features(incoming_df, scaler=scaler, fit=False)        
        results = run_drift_checks(model, ref_stats, incoming_engineered)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        results["timestamp"] = timestamp
        report_key = save_drift_report(results)

        if results["drift_detected"]:
            invoke_narration(results)
            #start_retraining(results)

        return {
            "statusCode": 200,
            "body": json.dumps({
                "drift_detected": results["drift_detected"],
                "triggered_by":  results["triggered_by"],
                "report_key":    report_key,
                "status":        "ok"
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e), "status": "error"})
        }