# Vigil 
Autonomous ML pipeline with drift detection & auto-retraining on AWS.

## Progress
- [x] S3 bucket + folder structure
- [x] IAM roles (Lambda, SageMaker, Step Functions)
- [x] SNS alerts
- [x] ECR repository
- [x] Data pipeline + baseline model
- [ ] Lambda functions (inference, monitoring, validation, promotion)
- [ ] CloudWatch metrics + alarms
- [ ] Step Functions retraining pipeline
- [ ] EventBridge scheduling
- [ ] GitHub Actions CI/CD
- [ ] Drift simulation + end-to-end test

## Project Structure
```text
vigil/
├── src/
│   ├── inference/       # Daily prediction Lambda
│   ├── monitoring/      # Drift metrics Lambda
│   ├── validation/      # Data validation Lambda
│   └── promotion/       # Model evaluation + promotion Lambda
├── ml/
│   ├── drift.py         # PSI + KS test logic
│   ├── features.py      # Feature engineering
│   └── train_local.py   # Local baseline training
├── infra/
│   ├── config.py        # ARNs + constants
│   ├── setup_s3.py      
│   ├── setup_iam.py      
│   ├── setup_sns.py
│   ├── setup_lambda.py
│   ├── setup_cloudwatch.py
│   ├── setup_stepfunctions.py
│   └── setup_eventbridge.py
├── statemachine/
│   └── retraining_pipeline.json
├── tests/
│   └── simulate_drift.py
├── data/
│   └── prepare_dataset.py
└── Dockerfile
```

## AWS Stack
| Service | Purpose |
|---|---|
| S3 | Data, models, predictions |
| IAM | Roles + permissions |
| Lambda | Inference + monitoring + orchestration |
| SageMaker | Training jobs |
| CloudWatch | Drift metrics + alarms |
| Step Functions | Retraining pipeline |
| EventBridge | Scheduling |
| SNS | Email alerts |
| ECR | Lambda container image |