import boto3
from botocore.exceptions import ClientError

client = boto3.client('ecr')


# ECR - Elastic Container Registry
# A container imagery service similar to docker, that stores all the contains, lambda's etc
# Lambda pulls the model (or anything for that matter) from the repository and runs it. 

def create_repository(name):
    try:
        response = client.create_repository(
            repositoryName = name
        )

        repo_uri = response['repository']['repositoryUri']
        print(f"Repository created successfully!\nRepositoryUri : {repo_uri}")

    except ClientError as e:
        print(f"Error creating repo: {e}")



if __name__ == "__main__":
    create_repository("vigil-lambda")  


    #Incase you want to delete a repo

        # response = client.delete_repository(
        #     repositoryName='vigil-lambda',
        #     force=True
        # )
        # print(response)