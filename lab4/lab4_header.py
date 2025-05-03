from datetime import datetime
import boto3
from botocore.config import Config

my_config = Config(
    region_name = 'eu-north-1'
)

# Get the service resource.

session = boto3.Session(
    aws_access_key_id='AKIAYAUOPXRYMHMYUQET',
    aws_secret_access_key='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'
)

dynamodb = session.resource('dynamodb', config=my_config)
scores_table = dynamodb.Table('IrisMinimal')
retrain_table = dynamodb.Table('IrisExtendedRetrain')