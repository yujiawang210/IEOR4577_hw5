import os
import json
import boto3
import time
import datetime
from pre_processing.pre_processing import PreProcessor

my_pre_processor = PreProcessor(padding_size=40, max_dictionary_size=10000)
sage_maker_client = boto3.client("runtime.sagemaker")


def lambda_handler(event, context):
    
    # Starting Time
    now = datetime.datetime.now()
    time1 = time.time()

    # Preprocessing
    tweet = event["tweet"]
    features = my_pre_processor.pre_process_text(tweet)

    time2 = time.time()

    # Sentiment Model
    model_payload = {
        "features_input": features
        }

    model_response = sage_maker_client.invoke_endpoint(
        EndpointName = "sentiment-model",
        ContentType = "application/json",
        Body = json.dumps(model_payload))

    result = json.loads(model_response["Body"].read().decode())

    time3 = time.time()

    # Model Response
    response = {}
    if result["predictions"][0][0] >= 0.5:
        response["sentiment"] = "positive"
    else:
        response["sentiment"] = "negative"

    print("Result: "+json.dumps(response, indent=2))

    # Payload Logging
    request_time = now.strftime("%b %d %Y %H:%M:%S")
    preprocessing_time = str(time2-time1)+"s"
    model_inference_time = str(time3-time2)+"s"

    logging = {
        "Request Time": request_time,
        "Tweet": event["tweet"], 
        "Sentiment": response["sentiment"], 
        "Probability": result["predictions"][0][0],
        "Preprocessing Time": preprocessing_time,
        "Model Inference Time": model_inference_time
        }

    log_object = json.dumps(logging)

    client = boto3.client("s3")
    filename = "request "+str(request_time)
    client.put_object(Body=log_object, Bucket='ieor4577-hw5', Key=filename+".json")

    # Return Response
    return response

