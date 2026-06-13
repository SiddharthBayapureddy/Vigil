# Stepfunctions - Orchestrator for the autonomour retraining pipeline. When monitoring Lambda detects drift, instead of calling 
# SageMaker + validation + promotion directly, it hands off to Step Functions which manages the entire flow with retries, error 
# handling, and state tracking.

# START
#   ↓
# LaunchSageMakerJob — trains challenger model
#   ↓ (wait for job to complete)
# ValidateChallenger — invoke vigil-validation Lambda
#   ↓
# Promote? 
#   ├── YES → PromoteChallenger — invoke vigil-promotion Lambda
#   │           ↓
#   │         END (success)
#   └── NO  → NotifyFailure — SNS: "Challenger rejected"
#               ↓
#             END (rejected)


import boto3
import json
from config import REGION, STEPFUNCTIONS_ROLE_ARN

sfn = boto3.client("stepfunctions", region_name=REGION)

MACHINE_NAME = "vigil-retraining-pipeline"


def create_or_update() -> str:
    
    with open("../statemachine/pipeline.json", "r") as f:
        definition = json.load(f)

    # Check if exists
    machines = sfn.list_state_machines()["stateMachines"]
    existing = next((m for m in machines if m["name"] == MACHINE_NAME), None)

    if existing:
        sfn.update_state_machine(
            stateMachineArn=existing["stateMachineArn"],
            definition=json.dumps(definition),
            roleArn=STEPFUNCTIONS_ROLE_ARN,
        )
        arn = existing["stateMachineArn"]
        print(f"Updated: {arn}")
    else:
        response = sfn.create_state_machine(
            name=MACHINE_NAME,
            definition=json.dumps(definition),
            roleArn=STEPFUNCTIONS_ROLE_ARN,
            type="STANDARD",
        )
        arn = response["stateMachineArn"]
        print(f"Created: {arn}")

    return arn


if __name__ == "__main__":
    arn = create_or_update()
    print(f"\nAdd to config.py:\nSTEPFUNCTIONS_ARN = \"{arn}\"")
    print("\nAlso update vigil-monitoring Lambda env var STEPFUNCTIONS_ARN")