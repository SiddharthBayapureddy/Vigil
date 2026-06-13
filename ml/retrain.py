# Vigil — retrain.py
# SageMaker training script — runs inside ECR container during automated retraining.
# Data mounted by SageMaker at SM_CHANNEL_TRAIN, model saved to SM_MODEL_DIR.

import json
import os

import boto3
import numpy as np
import pandas as pd
import xgboost as xgb

from features import build_features

# Sagemaker 

SM_CHANNEL_TRAIN = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
SM_MODEL_DIR     = os.environ.get("SM_MODEL_DIR",     "/opt/ml/model")

TRAIN_FILE = os.path.join(SM_CHANNEL_TRAIN, "train.csv")
VAL_FILE   = os.path.join(SM_CHANNEL_TRAIN, "validation.csv")
MODEL_FILE = os.path.join(SM_MODEL_DIR,     "model.json")

BUCKET           = "vigil-bucket"
CHAL_STATS_KEY   = "models/challenger/reference_stats.json"
SCALER_KEY       = "processed/scaler.pkl"

TARGET    = "RUL"
DROP_COLS = ["engine_id", "cycle", "RUL"]


s3 = boto3.client("s3", region_name="us-east-1")


def nasa_score(y_true, y_pred) -> float:
    diff = y_pred - y_true
    score = np.where(diff < 0,
                     np.exp(-diff / 13) - 1,
                     np.exp(diff  / 10) - 1)
    return float(np.sum(score))


def load() -> tuple:
    
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_FILE)
    val_df   = pd.read_csv(VAL_FILE)

    train_df["RUL"] = train_df["RUL"].clip(upper=125)

    X_train, y_train, scaler = build_features(train_df, fit=True, s3=s3)
    X_val,   y_val,   _      = build_features(val_df,   scaler=scaler)

    print(f"      Train: {X_train.shape} | Val: {X_val.shape}")
    return X_train, y_train, X_val, y_val


def train(X_train, y_train, X_val, y_val) -> xgb.XGBRegressor:
    print("Training...")
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.0,
        reg_lambda=1.0,
        random_state=67,
        tree_method="hist",
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
        early_stopping_rounds=20,
    )
    return model


def evaluate(model, X_val, y_val) -> dict:
    print("Evaluating...")
    y_pred = model.predict(X_val)
    mae    = float(np.mean(np.abs(y_val - y_pred)))
    mse    = float(np.mean((y_val - y_pred) ** 2))
    rmse   = float(np.sqrt(mse))
    ss_res = float(np.sum((y_val - y_pred) ** 2))
    ss_tot = float(np.sum((y_val - np.mean(y_val)) ** 2))
    r2     = float(1 - ss_res / ss_tot)
    nasa   = nasa_score(y_val.values, y_pred)

    print(f"      MAE  : {mae:.4f}")
    print(f"      RMSE : {rmse:.4f}")
    print(f"      R²   : {r2:.4f}")
    print(f"      NASA : {nasa:.4f}")

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

    return {
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


def save(model, stats) -> None:
    print("Saving...")
    os.makedirs(SM_MODEL_DIR, exist_ok=True)
    model.save_model(MODEL_FILE)
    print(f"      Model saved → {MODEL_FILE}")

    # Upload challenger stats to S3 — validation Lambda reads this
    s3.put_object(
        Bucket=BUCKET,
        Key=CHAL_STATS_KEY,
        Body=json.dumps(stats, indent=2)
    )
    print(f"      Stats → s3://{BUCKET}/{CHAL_STATS_KEY}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    X_train, y_train, X_val, y_val = load()
    model  = train(X_train, y_train, X_val, y_val)
    stats  = evaluate(model, X_val, y_val)
    save(model, stats)
    print("\nDone.")