# Triggered by sfns if challenger beats champion i.e., promotion = true
# Changes the model, archives current champion, and new challenger is moved to current/model.json
# ref_stats are modifed too
# Email alert sent regarding promotion

import json
import os
import sys
from datetime import datetime

import boto3
sys.path.append("/opt/python")

BUCKET          = "vigil-bucket"
CHAMPION_KEY    = "models/current/model.json"
CHALLENGER_KEY  = "models/challenger/model.json"
CHAL_STATS_KEY  = "models/challenger/reference_stats.json"
STATS_KEY       = "processed/reference_stats.json"
TOPIC_ARN       = os.environ.get("SNS_TOPIC_ARN")

s3 = boto3.client("s3")
sns = boto3.client("sns")


def archive_champion() -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    archive_key = f"models/archive/model_{timestamp}.json"
    s3.copy_object(
        Bucket = BUCKET,
        CopySource = {"Bucket":BUCKET , "Key":CHAMPION_KEY},
        Key = archive_key
    )

    return archive_key


def promote_challenger():

    # Model weights
    s3.copy_object(
        Bucket = BUCKET,
        CopySource = {"Bucket":BUCKET , "Key": CHALLENGER_KEY},
        Key = CHAMPION_KEY
    )

    # Ref stats
    s3.copy_object(
        Bucket=BUCKET,
        CopySource={"Bucket": BUCKET, "Key": CHAL_STATS_KEY},
        Key=STATS_KEY
    )


def publish_success(event: dict, archive_key: str) -> None:
    message = f"""Vigil — Model Promoted Successfully

    Challenger model has replaced the champion.

    Champion  MAE  : {event.get('champion_mae')}
    Challenger MAE : {event.get('challenger_mae')}

    Champion  NASA : {event.get('champion_nasa')}
    Challenger NASA: {event.get('challenger_nasa')}

    Old champion archived → s3://{BUCKET}/{archive_key}
    New champion live     → s3://{BUCKET}/{CHAMPION_KEY}
    """

    sns.publish(
        TopicArn=TOPIC_ARN,
        Subject="Vigil — Model Promoted",
        Message=message
    )


def lambda_handler(event, context):
    try:
        archive_key = archive_champion()
        promote_challenger()
        publish_success(event, archive_key)

        return {
            "promoted":       True,
            "archive_key":    archive_key,
            "champion_key":   CHAMPION_KEY,
            "champion_mae":   event.get("challenger_mae"),
            "champion_nasa":  event.get("challenger_nasa"),
            "status":         "ok"
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e), "status": "error"})
        }