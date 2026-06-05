import xgboost as xgb
import pandas as pd
import numpy as np
import json

import boto3

s3 = boto3.client("s3")
BUCKET = "vigil-bucket"
PATH = "raw-data/train/"
TRAIN_FILE = "train.csv"
VAL_FILE = "validation.csv"
WEIGHTS_FILE = "model_weights.json"


def load():
    df_train = pd.read_csv(TRAIN_FILE)
    X = df_train.drop(columns=["engine_id" , "cycle" , "RUL"])
    y = df_train["RUL"]
    return (X,y)


# Training on XGBoost
def train(X , y):

    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.0,
        reg_lambda=1.0,
        random_state=67
    )
    model.fit(X,y,
              eval_set = [(X,y)],
              verbose = True)
    
    return model

def evaluate(model , X):
    df_val = pd.read_csv(VAL_FILE)

    X_val = df_val.drop(columns=["engine_id" , "RUL" , "cycle"])
    y_val = df_val["RUL"]
    y_pred = model.predict(X_val)
    
    mae = np.mean(np.abs(y_val - y_pred))
    mse = np.mean((y_val - y_pred) ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_val - y_pred) ** 2)
    ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

    reference_stats = {
        "baseline_mae": float(mae),
        "feature_names": X_val.columns.tolist(),
        "n_samples": len(X)
    }

    return reference_stats



def save_and_upload_model(model , filename , s3_key , reference_stats):
    model.save_model(filename)
    s3.upload_file(filename , BUCKET , s3_key)

    # Saving baseline stats for drift detection
    with open("reference_stats.json" , "w") as file:
        json.dump(reference_stats,file)

    s3.upload_file("reference_stats.json" , BUCKET , "processed/reference_stats.json")


if __name__ == "__main__":

    # Download all datasets
    s3.download_file(BUCKET,PATH + "train.csv",TRAIN_FILE)
    s3.download_file(BUCKET,PATH + "validation.csv",VAL_FILE)

    X,y = load()
    model = train(X,y)

    ref_stats = evaluate(model,X)
    save_and_upload_model(model,WEIGHTS_FILE,"models/current/model_weights.json" , ref_stats)

