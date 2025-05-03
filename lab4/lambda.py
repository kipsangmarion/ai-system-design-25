import json
import logging
import boto3
import uuid
from boto3.dynamodb.types import TypeDeserializer

# Get the service resource.

client = boto3.client('dynamodb')
deserializer = TypeDeserializer()

def lambda_handler(event, context):
    # TODO implement

    print(event)
    for rec in event['Records']:
        print(rec)
        if rec['eventName'] == 'INSERT':
            UpdateItem = rec['dynamodb']['NewImage']
            print(UpdateItem)

            # lab4 code goes here
            # Convert from DynamoDB stream format to standard Python dict
            deserialized_item = {k: deserializer.deserialize(v) for k, v in UpdateItem.items()}
            print(deserialized_item)

            # Re-serialize it back for put_item
            marshalled_item = {
                k: {'S': str(v)} if isinstance(v, str) else
                   {'N': str(v)} if isinstance(v, (int, float)) else
                   {'BOOL': v} if isinstance(v, bool) else
                   {'NULL': True} if v is None else
                   {'S': json.dumps(v)}  # fallback for other types
                for k, v in deserialized_item.items()
            }
            marshalled_item['record_id'] = {'S': str(uuid.uuid4())}
            print(marshalled_item)

            if ("Class" in deserialized_item and 
                "Actual" in deserialized_item and 
                "Probability" in deserialized_item):
                
                if (deserialized_item["Class"] != deserialized_item["Actual"] or 
                    float(deserialized_item["Probability"]) < 0.9):
                    
                    response = client.put_item(TableName='IrisExtendedRetrain', Item=marshalled_item)
                    print(response)

    return {
        'statusCode': 200,
        'body': json.dumps( 'IrisExtendedRetrain Lambda return' )
    }
    
