import boto3
from botocore.exceptions import ClientError


BUCKET_NAME = "vigil-bucket" # Bucket name are globally unique across all AWS
REGION = "us-east-1"

PREFIXES = [
    "raw-data/train/",
    "raw-data/incoming/",
    "processed/",
    "models/current/",
    "models/candidates/",
    "predictions/",
    "drift-logs/",
]

def create_bucket():
    s3 = boto3.client("s3", region_name=REGION) # Opens a connection to S3 in us-east-1
    try:
        s3.create_bucket(Bucket=BUCKET_NAME) # Creates the bucket, no location constraint required, us-east-1 is default
        print(f"Bucket '{BUCKET_NAME}' created.")
    except ClientError as e:
        print(f"Bucket error: {e}")

def create_folders():
    s3 = boto3.client("s3")
    for prefix in PREFIXES:
        s3.put_object(Bucket=BUCKET_NAME, Key=prefix) # Creates folder
        print(f"Created folder: {prefix}")

if __name__ == "__main__":
    create_bucket()
    create_folders()
    print("S3 setup complete.")