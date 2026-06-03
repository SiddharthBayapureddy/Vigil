import boto3
import json
from botocore.exceptions import ClientError

# IAM = Identity and Access Management.
# Many services of AWS can't talk to each other by default. IAM is the system that grants permission.

# Specifying roles declares the stuff it can do, called policies. 

iam = boto3.client("iam")

# Trust Policies - tells AWS which service can assume this role

TRUST_POLICIES = {
    "lambda": json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }),
    "sagemaker": json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "sagemaker.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }),
    "stepfunctions": json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "states.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }),
}

# Roles and their policies
ROLES = {
    "vigil-lambda-role": {
        "trust": "lambda",
        "policies": [
            "arn:aws:iam::aws:policy/AmazonS3FullAccess",
            "arn:aws:iam::aws:policy/CloudWatchFullAccess",
            "arn:aws:iam::aws:policy/AmazonSNSFullAccess",
            "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
            "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
        ]
    },
    "vigil-sagemaker-role": {
        "trust": "sagemaker",
        "policies": [
            "arn:aws:iam::aws:policy/AmazonS3FullAccess",
            "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
        ]
    },
    "vigil-stepfunctions-role": {
        "trust": "stepfunctions",
        "policies": [
            "arn:aws:iam::aws:policy/AWSLambda_FullAccess",
            "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
            "arn:aws:iam::aws:policy/AmazonSNSFullAccess",
        ]
    },
}

def create_role(name,key):

    try:
        response = iam.create_role(
            RoleName = name,
            AssumeRolePolicyDocument = TRUST_POLICIES[key]
        )

    except ClientError as e:
        if e.response["Error"]["Code"] == "EntityAlreadyExists":
            print(f"Role already exists: {name}")
            return iam.get_role(RoleName=name)["Role"]["Arn"]
        raise


def attach_policies(role_name, policies):
    for policy_arn in policies:
        iam.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
        print(f"  Attached: {policy_arn.split('/')[-1]}")


if __name__ == "__main__":
    arns = {}
    for role_name, config in ROLES.items():
        arn = create_role(role_name, config["trust"])
        attach_policies(role_name, config["policies"])
        arns[role_name] = arn
        print()

    print("IAM setup complete.")
    print("\nRole ARNs (save these):")
    for name, arn in arns.items():
        print(f"  {name}: {arn}")