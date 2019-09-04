import aih
import base64
import json

aih_client = aih.init(app_id="APP_ID", secret_key="SECRET_KEY")


def detect_face(event, context):
    try:
        body = json.loads(event["body"])
        file = body["file"]
        binary_data_file = base64.b64decode(file)
        res = aih_client.detect_face(file_binary=binary_data_file)
        return aih.response(res)
    except Exception as e:
        exception_type = e.__class__.__name__
        exception_message = str(e)
        api_exception_obj = {
            "isError": True,
            "type": exception_type,
            "message": exception_message
        }
        api_exception_json = json.dumps(api_exception_obj)
        raise aih.LambdaException(api_exception_json)


def upload_faces(event, context):
    try:
        body = json.loads(event["body"])
        fullname = body["fullname"]
        files = body["files"]
        images = []
        for file in files:
            images.append(base64.b64decode(file))
        res = aih_client.upload_images(fullname=fullname, images=images)
        return aih.response(res)
    except Exception as e:
        exception_type = e.__class__.__name__
        exception_message = str(e)
        api_exception_obj = {
            "isError": True,
            "type": exception_type,
            "message": exception_message
        }
        # Create a JSON string
        api_exception_json = json.dumps(api_exception_obj)
        raise aih.LambdaException(api_exception_json)


def start_training(event, context):
    try:
        body = json.loads(event["body"]) if event["body"] else {}
        instance_type = body["instance_type"] if 'instance_type' in body else None
        res = aih_client.train_model(instance_type=instance_type)
        return aih.response(res)
    except Exception as e:
        exception_type = e.__class__.__name__
        exception_message = str(e)
        api_exception_obj = {
            "isError": True,
            "type": exception_type,
            "message": exception_message
        }
        # Create a JSON string
        api_exception_json = json.dumps(api_exception_obj)
        raise aih.LambdaException(api_exception_json)

def get_models(event, context):
    try:
        res = aih_client.list_models()
        return aih.response(res)
    except Exception as e:
        exception_type = e.__class__.__name__
        exception_message = str(e)
        api_exception_obj = {
            "isError": True,
            "type": exception_type,
            "message": exception_message
        }
        # Create a JSON string
        api_exception_json = json.dumps(api_exception_obj)
        raise aih.LambdaException(api_exception_json)


def get_model(event, context):
    try:
        model_id = event["pathParameters"]["modelId"]
        res = aih_client.get_model(model_id=model_id)
        return aih.response(res)
    except Exception as e:
        exception_type = e.__class__.__name__
        exception_message = str(e)
        api_exception_obj = {
            "isError": True,
            "type": exception_type,
            "message": exception_message
        }
        # Create a JSON string
        api_exception_json = json.dumps(api_exception_obj)
        raise aih.LambdaException(api_exception_json)


def stop_training(event, context):
    try:
        model_id = event["pathParameters"]["modelId"]
        res = aih_client.stop_training(model_id=model_id)
        return aih.response(res)
    except Exception as e:
        exception_type = e.__class__.__name__
        exception_message = str(e)
        api_exception_obj = {
            "isError": True,
            "type": exception_type,
            "message": exception_message
        }
        api_exception_json = json.dumps(api_exception_obj)
        raise aih.LambdaException(api_exception_json)


def polly_speak(event, context):
    try:
        text = json.loads(event["body"])["text"]
        res = aih_client.speak(text=text)
        return aih.response(res)
    except Exception as e:
        exception_type = e.__class__.__name__
        exception_message = str(e)
        api_exception_obj = {
            "isError": True,
            "type": exception_type,
            "message": exception_message
        }
        api_exception_json = json.dumps(api_exception_obj)
        raise aih.LambdaException(api_exception_json)


def check_untrain_images(event, context):
    try:
        res = aih_client.is_exists_untrain_images()
        return aih.response(res)
    except Exception as e:
        exception_type = e.__class__.__name__
        exception_message = str(e)
        api_exception_obj = {
            "isError": True,
            "type": exception_type,
            "message": exception_message
        }
        # Create a JSON string
        api_exception_json = json.dumps(api_exception_obj)
        raise aih.LambdaException(api_exception_json)
