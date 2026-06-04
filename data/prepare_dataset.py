import pandas as pd
import boto3
from botocore.exceptions import ClientError
import os

s3 = boto3.client("s3")
BUCKET_NAME = "vigil-bucket"

columns = ["engine_id", "cycle"] + \
        [f"op_{i}" for i in range(1, 4)] + \
        [f"sensor_{i}" for i in range(1, 22)]

def load_train():
    df = pd.read_csv("raw/train_FD001.txt" , sep = " " , header = None)
    df = df.drop(columns = [26,27])
    df.columns = columns
    df['RUL'] = df.groupby("engine_id")["cycle"].transform("max") - df["cycle"]
    
    return df


def load_test():
    df_test = pd.read_csv("raw/test_FD001.txt" , sep=" ", header = None)
    df_test = df_test.drop(columns = [26,27])
    df_test.columns = columns

    df_rul = pd.read_csv("raw/RUL_FD001.txt" , header = None , names = ["RUL_at_end"])
    df_rul["engine_id"] = df_rul.index + 1

    df_test = df_test.merge(df_rul,on="engine_id")
    df_test["RUL"] = df_test["RUL_at_end"] + \
                 (df_test.groupby("engine_id")["cycle"].transform("max") - df_test["cycle"])
    df_test.drop(columns=["RUL_at_end"], inplace=True)
        
    return df_test


def upload_to_s3(path , s3_key):
    s3.upload_file(path,BUCKET_NAME,s3_key)
    print(f"Uploaded {path} to s3://{BUCKET_NAME}/{s3_key}")

def split_and_upload(train_df , test_df):
    
    train = train_df[train_df["engine_id"] <= 80]
    val = train_df[train_df["engine_id"] > 80]
    test = test_df

    train.to_csv("train.csv" , index = False)
    test.to_csv("test.csv" , index = False)
    val.to_csv("val.csv" , index = False)

    upload_to_s3("train.csv", "raw-data/train/train.csv")
    upload_to_s3("val.csv", "raw-data/train/validation.csv")
    upload_to_s3("test.csv", "raw-data/incoming/test.csv")



if __name__ == "__main__":
    train = load_train()
    test = load_test()
    split_and_upload(train,test)