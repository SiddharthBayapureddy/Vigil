# Lambda setup script

import boto3
import json
import os
import time


from config import (
    ACCOUNT_ID,
    REGION,
    LAMBDA_ROLE_ARN,
    SNS_TOPIC_ARN,
)


MISTRAL_API_KEY = os.environ["MISTRAL_API_KEY"]
ECR_IMAGE   = f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/vigil-lambda:latest"
TIMEOUT     = 300 #5min
MEMORY      = 512 #512MB

lam = boto3.client("lambda", region_name=REGION)

FUNCTIONS = [
    {
        "name":    "vigil-inference",
        "handler": "inference.handler.lambda_handler",
        "env": {},
    },
    {
        "name":    "vigil-monitoring",
        "handler": "monitoring.handler.lambda_handler",
        "env": {
            "STEPFUNCTIONS_ARN": os.environ.get("STEPFUNCTIONS_ARN" , ""),
        },
    },
    {
        "name":    "vigil-narration",
        "handler": "narration.handler.lambda_handler",
        "env": {
            "SNS_TOPIC_ARN":    SNS_TOPIC_ARN,
            "MISTRAL_API_KEY":  MISTRAL_API_KEY
        },
    },
    {
        "name":    "vigil-validation",
        "handler": "validation.handler.lambda_handler",
        "env": {},
    },
    {
        "name":    "vigil-promotion",
        "handler": "promotion.handler.lambda_handler",
        "env": {
            "SNS_TOPIC_ARN": SNS_TOPIC_ARN,
        },
    },
]


def create_or_update(fn: dict) -> None:
    name    = fn["name"]
    handler = fn["handler"]
    env     = fn["env"]

    try:
        lam.create_function(
            FunctionName=name,
            PackageType="Image",
            Code={"ImageUri": ECR_IMAGE},
            ImageConfig={"Command": [handler]},
            Role=LAMBDA_ROLE_ARN,
            Timeout=TIMEOUT,
            MemorySize=MEMORY,
            Environment={"Variables": env},
        )
        print(f"✅ Created  : {name}")

    except lam.exceptions.ResourceConflictException:
        # already exists — update code + config
        lam.update_function_code(
            FunctionName=name,
            ImageUri=ECR_IMAGE,
        )

        # Wait for code update to complete
        waiter = lam.get_waiter("function_updated")
        waiter.wait(FunctionName=name)

        lam.update_function_configuration(
            FunctionName=name,
            ImageConfig={"Command": [handler]},
            Timeout=TIMEOUT,
            MemorySize=MEMORY,
            Environment={"Variables": env},
        )
        print(f"🔄 Updated  : {name}")


if __name__ == "__main__":
    for fn in FUNCTIONS:
        create_or_update(fn)

    print("\nDone!")