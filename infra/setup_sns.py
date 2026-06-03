import boto3
from botocore.exceptions import ClientError
from config import REGION

client = boto3.client('sns', region_name = REGION)

def create_topic():
    try:
        response = client.create_topic(Name = "vigil-alerts")
        topic_arn = response["TopicArn"]
        print(f"Created Topic: {topic_arn}")
        return topic_arn

    except ClientError as e:
        print(f"Topic error: {e}")


def subscribe_email(topic_arn,email):

    try:
        client.subscribe(
            TopicArn = topic_arn,
            Protocol = "email",
            Endpoint = email
        )
        print(f"Confirmation Mail sent to {email}!")

    except ClientError as e:
        print(f"Subscription Error: {e}")


if __name__ == "__main__":
    topic_arn = create_topic()
    subscribe_email(topic_arn,"siddharthbayapureddy@gmail.com")
    print(f"\nSave this ARN to config.py:\nSNS_TOPIC_ARN = \"{topic_arn}\"")