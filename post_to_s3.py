# Takes in scraped information and sends it to an s3 bucket
import boto3
import pandas as pd
import os


def get_client_bucket():
    ACCESS_KEY = os.environ['AWS_ACCESS_KEY_ID']
    SECRET_KEY = os.environ['AWS_SECRET_ACCESS_KEY']
    boto3_connection = boto3.resource('s3',
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
    )
    bucket = boto3_connection.Bucket('peterrussodsiproj')
    return boto3_connection, bucket

def create_new(start_df):
    boto3_connection, bucket = get_client_bucket()
    start_df.to_csv('temp_data.csv')
    bucket.Object('data.csv').put('temp_data.csv')


def Add_New(adding_df):
    boto3_connection, bucket = get_client_bucket()
    obj = bucket.Object('data.csv')
    df = pd.from_csv(obj)
    temp_cols = df.columns.copy()
    df = df.append(adding_df)
    for col in df.columns:
        if col not in temp_cols:
            df.drop(col, inplace=True, axis=1)
    df.to_csv('temp_data.csv')
    new_bucket.Object('data.csv').put('temp_data.csv')
