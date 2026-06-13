# Vigil

An autonomous MLOps pipeline for predictive maintenance of aircraft engines. Vigil monitors a live XGBoost model for data drift, generates plain-English incident reports via LLM, and automatically retrains and promotes a challenger model — entirely serverless on AWS.

Built on NASA's CMAPSS FD001 dataset. Predicts Remaining Useful Life (RUL) of turbofan engines in cycles.

---

## Architecture

```
EventBridge (daily cron)
        │
        ▼
┌─────────────────────┐
│  Monitoring Lambda  │  Runs PSI + KS drift detection on incoming sensor data
│                     │  Saves drift report to S3
└────────┬────────────┘
    drift│detected
         ▼
┌─────────────────────┐
│  Narration Lambda   │  Calls Mistral API → plain-English incident report
│                     │  Publishes alert via SNS
└────────┬────────────┘
         ▼
┌─────────────────────────────────────────────────────────┐
│  Step Functions — Retraining Pipeline                   │
│                                                         │
│  SageMaker Training Job                                 │
│       → trains challenger model on latest data          │
│       ↓                                                 │
│  Validation Lambda                                      │
│       → champion vs challenger (MAE + NASA score)       │
│       ↓                                                 │
│  Promote?                                               │
│       ├── YES → Promotion Lambda → swap model in S3     │
│       └── NO  → SNS alert, keep champion                │
└─────────────────────────────────────────────────────────┘

API Gateway → Inference Lambda → predict RUL → CloudWatch metric
```

---

## What It Does

- Trains an XGBoost regressor on CMAPSS FD001 with rolling window feature engineering
- Detects distributional shift in production sensor data using PSI and KS tests
- Generates human-readable drift incident reports using Mistral LLM
- Autonomously retrains a challenger model via SageMaker when drift is confirmed
- Validates the challenger against the champion before any promotion
- Serves RUL predictions via a REST endpoint backed by API Gateway + Lambda
- Logs all predictions as CloudWatch custom metrics

No manual intervention required once deployed. The entire loop — detect, narrate, retrain, validate, promote — runs autonomously on a daily schedule.

---

## Model Performance

| Metric | Baseline (raw features) | Final (engineered features) |
|--------|------------------------|------------------------------|
| MAE    | 36.53 cycles           | 9.277 cycles                 |
| RMSE   | 51.08                  | 12.82                        |
| R²     | 0.565                  | 0.9035                       |

Feature engineering improvements over raw sensors:
- Rolling mean, std, min, max over 5 and 10 cycle windows per engine
- Delta from each engine's own cycle-1 baseline (captures absolute degradation)
- Normalized cycle position (0.0 → 1.0) fixed to training set max cycles
- RUL clipping at 125 cycles (standard CMAPSS practice)
- NASA asymmetric scoring penalizing late predictions

---

## Drift Detection

Three independent checks run on every incoming batch:

**Population Stability Index (PSI)**
Measures distributional shift between training and production data per feature.
Threshold: PSI > 0.2 flags drift.

```
PSI = Σ (actual% - expected%) × ln(actual% / expected%)
```

- PSI < 0.1 — no meaningful change
- 0.1 to 0.2 — moderate shift, monitor
- > 0.2 — significant drift, trigger retraining

**Kolmogorov-Smirnov Test**
Non-parametric test comparing whether two samples come from the same distribution.
Threshold: KS statistic > 0.1.

**MAE Degradation**
Compares current model MAE on incoming data against baseline.
Threshold: ratio > 1.5x baseline MAE.

Drift is declared if any single check fires. The full drift report is saved to S3 under `processed/drift_reports/` with timestamp for auditability.

---

## Stack

| Layer | Technology |
|-------|-----------|
| Model | XGBoost Regressor |
| Feature Engineering | pandas, numpy, scikit-learn |
| Drift Detection | PSI, KS test (scipy) |
| LLM Narration | Mistral API (mistral-small-latest) |
| Serving | AWS Lambda + API Gateway |
| Orchestration | AWS Step Functions |
| Retraining | AWS SageMaker Training Jobs |
| Scheduling | AWS EventBridge |
| Alerting | AWS SNS |
| Storage | AWS S3 |
| Observability | AWS CloudWatch |
| Containerization | Docker (ECR) |
| CI/CD | GitHub Actions |
| IaC | boto3 (Python) |

---

## Project Structure

```
Vigil/
├── infra/
│   ├── config.py                  ARNs and constants
│   ├── setup_s3.py                S3 bucket + folder structure
│   ├── setup_iam.py               IAM roles for Lambda, SageMaker, Step Functions
│   ├── setup_sns.py               SNS topic + email subscription
│   ├── setup_ecr.py               ECR repository
│   ├── setup_lambda.py            Deploy all 5 Lambda functions from ECR
│   ├── setup_cloudwatch.py        Dashboard + alarms
│   ├── setup_stepfunctions.py     Retraining state machine
│   └── setup_eventbridge.py       Daily cron trigger
│
├── ml/
│   ├── features.py                Rolling window feature engineering + scaler
│   ├── train_local.py             Local training + S3 upload
│   ├── drift.py                   PSI + KS + MAE drift checks
│   └── retrain.py                 SageMaker training entrypoint
│
├── src/
│   ├── inference/handler.py       Predict RUL, emit CloudWatch metric
│   ├── monitoring/handler.py      Run drift checks, trigger pipeline
│   ├── narration/handler.py       Mistral API → SNS alert
│   ├── validation/handler.py      Champion vs challenger comparison
│   └── promotion/handler.py       Swap challenger → champion in S3
│
├── statemachine/
│   └── pipeline.json              Step Functions ASL definition
│
├── tests/
│   └── simulate_drift.py          End-to-end drift simulation test
│
├── .github/
│   └── workflows/deploy.yml       CI/CD pipeline
│
├── Dockerfile                     Single image for all Lambda handlers + SageMaker
└── requirements.txt
```

---

## S3 Layout

```
vigil-bucket/
├── raw-data/
│   ├── train/
│   │   ├── train.csv              Engines 1-80
│   │   └── validation.csv         Engines 81-100
│   └── incoming/
│       └── test.csv               Production batch (drift simulation target)
├── models/
│   ├── current/
│   │   └── model.json             Live champion model
│   ├── challenger/
│   │   ├── model.json             Retrained challenger (pre-promotion)
│   │   └── reference_stats.json   Challenger baseline stats
│   └── archive/
│       └── model_{timestamp}.json Versioned model history
└── processed/
    ├── reference_stats.json       Champion baseline stats + feature distributions
    ├── scaler.pkl                 Fitted StandardScaler
    └── drift_reports/
        └── report_{timestamp}.json  Full drift report per run
```

---

## Setup

### Prerequisites

- AWS account with CLI configured (`aws configure`)
- Docker Desktop
- Python 3.11+
- Mistral API key

### Environment Variables

Create a `.env` file (never commit this):

```
MISTRAL_API_KEY=your_key_here
SNS_TOPIC_ARN=arn:aws:sns:us-east-1:YOUR_ACCOUNT:vigil-alerts
STEPFUNCTIONS_ARN=arn:aws:states:us-east-1:YOUR_ACCOUNT:stateMachine:vigil-retraining-pipeline
```

### Deploy Infrastructure

```bash
cd infra
python setup_s3.py
python setup_iam.py
python setup_sns.py        # confirm subscription email
python setup_ecr.py
```

### Train Baseline Model

```bash
cd ml
pip install -r requirements.txt
python train_local.py
```

### Build and Push Docker Image

```bash
docker buildx build --platform linux/amd64 --push \
  -t YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/vigil-lambda:latest .
```

### Deploy Lambdas and Orchestration

```bash
cd infra

# Set env vars first
export STEPFUNCTIONS_ARN=...
export SNS_TOPIC_ARN=...
export MISTRAL_API_KEY=...

python setup_lambda.py
python setup_stepfunctions.py
python setup_eventbridge.py
python setup_cloudwatch.py
```

---

## Testing

### Test Inference

```bash
aws lambda invoke \
  --function-name vigil-inference \
  --payload '{
    "engine_id": 1,
    "cycle": 150,
    "sensors": {
      "op_1": 0.0, "op_2": 0.0, "op_3": 0.0,
      "sensor_1": 0.0, "sensor_2": 641.82, "sensor_3": 1589.7,
      "sensor_4": 1400.6, "sensor_5": 0.0, "sensor_6": 0.0,
      "sensor_7": 554.36, "sensor_8": 2388.06, "sensor_9": 9065.25,
      "sensor_10": 0.0, "sensor_11": 47.47, "sensor_12": 521.66,
      "sensor_13": 2388.06, "sensor_14": 8138.62, "sensor_15": 8.4195,
      "sensor_16": 0.0, "sensor_17": 392.0, "sensor_18": 0.0,
      "sensor_19": 0.0, "sensor_20": 38.86, "sensor_21": 23.3735
    }
  }' \
  --cli-binary-format raw-in-base64-out \
  response.json && cat response.json
```

### Trigger Drift Detection Manually

```bash
aws lambda invoke \
  --function-name vigil-monitoring \
  --payload '{}' \
  --cli-binary-format raw-in-base64-out \
  response.json && cat response.json
```

Check Step Functions for retraining execution:

```bash
aws stepfunctions list-executions \
  --state-machine-arn YOUR_STATE_MACHINE_ARN \
  --query "executions[0]"
```

---

## Autonomous Loop — How It Works End to End

1. EventBridge fires daily at midnight UTC
2. Monitoring Lambda downloads incoming sensor batch from S3
3. Feature engineering applied (same pipeline as training)
4. PSI + KS tests run against training distribution stored in `reference_stats.json`
5. If drift detected:
   - Drift report saved to S3 with timestamp
   - Narration Lambda called async → Mistral generates incident report → SNS email
   - Step Functions execution started
6. Step Functions: SageMaker launches training job on `ml.m5.large`
   - Pulls `train.csv` + `validation.csv` from S3
   - Runs `retrain.py` inside ECR container
   - Saves challenger model to `models/challenger/`
7. Validation Lambda: loads champion + challenger, evaluates both on val set
   - Challenger must beat champion on both MAE and NASA score to pass
8. If challenger wins: Promotion Lambda archives champion, copies challenger to `models/current/`
9. SNS alert sent with promotion metrics or rejection reason

---

## Champion / Challenger Pattern

Every retrain produces a challenger model evaluated against the live champion before any promotion. This prevents a degraded retrain from going live.

Challenger must beat champion on:
- MAE (mean absolute error in RUL cycles)
- NASA asymmetric score (penalizes late predictions more heavily than early)

If challenger fails either metric, champion stays live and a rejection alert is sent.

---

## Dataset

NASA CMAPSS FD001 — turbofan engine degradation simulation dataset.

- 100 engines, single operational condition, one fault mode
- Training set: engines 1-80 (~16,000 rows)
- Validation set: engines 81-100 (~4,500 rows)
- Test set: used as incoming production batch for drift simulation
- 21 sensors + 3 operational settings per row
- Target: RUL in cycles (clipped at 125 for training stability)

Reference: Saxena, A. et al. "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation." ISTED, 2008.

---

## Cost

Designed to run within AWS Free Tier limits.

| Service | Usage | Cost |
|---------|-------|------|
| Lambda | 5 functions, invoked daily | Free tier |
| S3 | ~50MB storage | Free tier |
| ECR | 1 image ~1.2GB | Free tier (500MB/month) |
| CloudWatch | Custom metrics + logs | Free tier |
| Step Functions | ~30 state transitions/month | Free tier |
| EventBridge | 1 rule, daily | Free tier |
| SNS | <1000 emails/month | Free tier |
| SageMaker | ml.m5.large training job | Not free tier — ~$0.115/hour per run |

SageMaker is the only billable component. Each retraining job runs for approximately 5-10 minutes — cost per retraining run is under $0.02.

---

## License

MIT