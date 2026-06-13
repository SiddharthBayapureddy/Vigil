# Inputs the drift report from monitoring, parses via an LLM and sends the final natural language report to admin via sns

import json
import os
import sys

import boto3

from mistralai.client import Mistral
MODEL = "mistral-medium-latest"

TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN")
api_key = os.environ.get("MISTRAL_API_KEY")

sns = boto3.client("sns")
client = Mistral(api_key = api_key)


def build_prompt(report: dict) -> str:

    return f"""You are an MLOps monitoring assistant for Vigil, an autonomous predictive maintenance system for aircraft engines.

A drift detection check just fired. Here is the raw drift report:
{json.dumps(report, indent=2)}

Write a concise incident report for an on-call engineer covering:
1. What drifted and how severely (reference specific sensors and PSI/KS scores)
2. Whether model performance has degraded (MAE ratio)
3. What automated actions have been triggered
4. Recommended follow-up action

Keep it under 150 words. Be direct. No fluff.
Use plain text only — no markdown, no asterisks, no bullet symbols. Use numbered lists and plain punctuation only."""


def narrate(prompt: str):

    chat_response = client.chat.complete(
        model = MODEL,
        messages = [
            {
                "role": "user",
                "content": prompt,
            },
        ]
    )

    message = chat_response.choices[0].message

    if message is None:
        return ""
    
    content = message.content

    if content is None:
        return ""
    
    return str(content).strip()


def publish_alert(narration: str, drift_detected: bool) -> None:

    subject = ""
    if drift_detected:
        subject = "Vigil - Drift Detected"
    else:
        subject = "Vigil - No Drift Detected"

    sns.publish(
        TopicArn=TOPIC_ARN,
        Subject=subject,
        Message=narration
    )


def lambda_handler(event,context):

    try:
        prompt = build_prompt(event)
        narration = narrate(prompt)
        publish_alert(narration,event.get("drift_detected" , False))

        return {
            "statusCode": 200,
            "body": json.dumps({
                "narration": narration,
                "status":    "ok"
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e), "status": "error"})
        }
