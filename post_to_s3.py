# Takes in scraped information and sends it to an s3 bucket
import boto3
import pandas as pd


def create_new(start_df):
    boto3_connection = boto3.resource('s3')
    bucket = boto3_connection.Bucket('peterrussodsiproj')
    start_df.to_csv('temp_data.csv')
    bucket.Object('data.csv').put('temp_data.csv')


def Add_New(adding_df):
    boto3_connection = boto3.resource('s3')
    bucket = boto3_connection.Bucket('peterrussodsiproj')
    obj = bucket.Object('data.csv')
    df = pd.from_csv(obj)
    df = pd.concat([df, adding_df])
    df.to_csv('temp_data.csv')
    new_bucket.Object('data.csv').put('temp_data.csv')
