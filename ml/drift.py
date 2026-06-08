import boto3
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import xgboost as xgb

BUCKET = "vigil-bucket"
INCOMING_KEY = "raw-data/incoming/test.csv"
INCOMING_FILE = "test.csv"
STATS_KEY = "processed/reference_stats.json"
STATS_FILE = "reference_stats.json"

MODEL_FILE = "model.json"
MODEL_KEY = "models/current/model.json"

DROP_COLS = ["engine_id", "cycle", "RUL"]
TARGET    = "RUL"


# Determining whether drift exists or not
PSI_THRESHOLD = 0.2
KS_THRESHOLD  = 0.1
MAE_THRESHOLD = 1.5


# Population Stability Index (PSI) - 
# PSI = Σ (actual% - expected%) × ln(actual% / expected%)
# Used to check data drift between two samples - Actual & Expected
# Expected refers to the baseline underlying distribution of data and how it's expected to be, and Actual refers to the true real data

# PSI < 0.1 - Little/no change
# 0.1 < PSI < 0.25 - Moderate Shift
# PSI > 0.25 - Large Shift

def compute_psi(ref_counts , incoming_vals , bins) -> float:
    '''Calculate PSI from reference_stats and the incoming values and outputs it'''

    ref_counts = np.array(ref_counts,dtype=float)
    incoming_cnts , _ = np.histogram(incoming_vals , bins = bins)
    incoming_cnts = incoming_cnts.astype(float)

    ref_pct = ref_counts/ref_counts.sum()
    incoming_pct = incoming_cnts/incoming_cnts.sum()

    eps = 1e-10
    ref_pct = np.where(ref_pct == 0 , eps  , ref_pct)
    incoming_pct = np.where(incoming_pct == 0 , eps , incoming_pct)

    psi = float(np.sum((incoming_pct - ref_pct) * np.log(incoming_pct / ref_pct)))
    return psi



# Kolmogorov-Smirnov Test (KS Test) 
# D = max​∣F1​(x)−F2​(x)∣
# Used to check whether two distributions are different 
def compute_ks(ref_mean, ref_std, incoming_vals) -> tuple:
    '''Calculate KS from two distributions'''

    ref_vals = np.random.normal(loc=ref_mean, scale=ref_std, size=1000)
    stat, p_value = ks_2samp(ref_vals, incoming_vals)

    return float(stat), float(p_value)


# Checking if avg mae is significantly more than baseline mae
def compute_mae_ratio(model : xgb.XGBRegressor , incoming_X, incoming_y, baseline_mae) -> float:

    y_pred = model.predict(incoming_X)
    incoming_mae = float(np.mean(np.abs(incoming_y - y_pred)))
    return incoming_mae / baseline_mae



def download_stats(s3 : boto3.client):
    s3.download_file(BUCKET,STATS_KEY, STATS_FILE)
    s3.download_file(BUCKET,INCOMING_KEY,INCOMING_FILE)
    s3.download_file(BUCKET,MODEL_KEY,MODEL_FILE)

    incoming_val = pd.read_csv(INCOMING_FILE)
    
    reference_stats = {}
    with open(STATS_FILE , "r") as file:
        reference_stats = json.load(file)

    model = xgb.XGBRegressor()
    model.load_model(MODEL_FILE)

    return incoming_val, reference_stats, model


def run_drift_checks(model : xgb.XGBRegressor , reference_stats : dict , incoming_val : pd.DataFrame):
 
    psi_scores = {}
    ks_scores  = {}
 
    for feat in reference_stats["feature_names"]:
        ref_counts    = reference_stats["features"][feat]["hist_counts"]
        bins          = reference_stats["features"][feat]["hist_bins"]
        ref_mean      = reference_stats["features"][feat]["mean"]
        ref_std       = reference_stats["features"][feat]["std"]
        incoming_vals = incoming_val[feat].values

        
 
        psi_scores[feat]      = compute_psi(ref_counts, incoming_vals, bins)
        ks_stat, ks_p         = compute_ks(ref_mean, ref_std, incoming_vals)
        ks_scores[feat]       = {"stat": ks_stat, "p_value": ks_p}
 
    incoming_X = incoming_val.drop(columns=DROP_COLS, errors="ignore")
    incoming_y = incoming_val[TARGET]
    mae_ratio  = compute_mae_ratio(model, incoming_X, incoming_y,
                                   reference_stats["baseline_mae"])
 
    triggered_by = []
    for feat, psi in psi_scores.items():
        if psi > PSI_THRESHOLD:
            triggered_by.append(f"PSI:{feat}")
    for feat, ks in ks_scores.items():
        if ks["stat"] > KS_THRESHOLD:
            triggered_by.append(f"KS:{feat}")
    if mae_ratio > MAE_THRESHOLD:
        triggered_by.append("MAE")
 
    return {
        "drift_detected": len(triggered_by) > 0,
        "mae_ratio":      round(mae_ratio, 4),
        "psi_scores":     {k: round(v, 4) for k, v in psi_scores.items()},
        "ks_scores":      {k: {"stat":    round(v["stat"],    4),
                               "p_value": round(v["p_value"], 4)}
                           for k, v in ks_scores.items()},
        "triggered_by":   triggered_by,
    }
 


if __name__ == "__main__":

    s3 = boto3.client("s3")
    incoming_val,ref_stats, model= download_stats(s3)
    results = run_drift_checks(model, ref_stats, incoming_val)

    print("\n── Drift Report ──────────────────────────────")
    print(f"Drift Detected : {results['drift_detected']}")
    print(f"MAE Ratio      : {results['mae_ratio']}x  (threshold: {MAE_THRESHOLD}x)")
    print(f"Triggered By   : {results['triggered_by'] or 'None'}")
 
    print("\nPSI Scores (threshold > 0.2):")
    for feat, psi in results["psi_scores"].items():
        flag = " ⚠" if psi > PSI_THRESHOLD else ""
        print(f"  {feat:<15} {psi:.4f}{flag}")
 
    print("\nKS Stats (threshold > 0.1):")
    for feat, ks in results["ks_scores"].items():
        flag = " ⚠" if ks["stat"] > KS_THRESHOLD else ""
        print(f"  {feat:<15} stat={ks['stat']:.4f}  p={ks['p_value']:.4f}{flag}")



    # Cleaning up
    for file in [MODEL_FILE,STATS_FILE, INCOMING_FILE]:
        if(os.path.exists(file)):
            os.remove(file)
        
    
