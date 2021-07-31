from flask import Blueprint, Response
from flask import request, jsonify
import os
import cv2 
import numpy as np
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder

api_version = os.environ['API_VERSION']
image_width = int(os.environ['IMAGE_WIDTH'])
image_height = int(os.environ['IMAGE_HEIGHT'])
lpd_threshold = float(os.environ['LPD_THRESHOLD'])
images_path = '/root/app/images'
checkpoint = '/root/app/models/lpd_model/ckpt'
pipeline_config = '/root/app/models/lpd_model/pipeline.config'

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(pipeline_config)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(checkpoint).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

# Make a prediction to load cache and save time on following requests
img = cv2.imread(os.path.join(images_path, 'test-0001.jpg'))
image_np = np.array(img)
input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
detections = detect_fn(input_tensor)

lpr = Blueprint('lpr', __name__, template_folder='templates')

def license_plate_detection(request, result):
    try:
        image_filename = request.json['filename']
    except:
        result['statusCode'] = 400
        result['errorMessage'] = 'Wrong request body'
        return result

    image = cv2.imread(os.path.join(images_path, image_filename))
    if image is None:
        result['statusCode'] = 500
        result['errorMessage'] = 'Image file not found'
        return result

    image_np = np.array(image)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    
    result['detectionScore'] = str(detections['detection_scores'][0])
    result['lowDetectionScore'] = True if detections['detection_scores'][0] < lpd_threshold else False
    result['detectionBox'] = [
        int(detections['detection_boxes'][0][0]*image_height), 
        int(detections['detection_boxes'][0][1]*image_width),
        int(detections['detection_boxes'][0][2]*image_height),
        int(detections['detection_boxes'][0][3]*image_width)
    ]

    return result

@lpr.route(f'/api/{api_version}/lpr/predict', methods=['GET'])
def lpr_predict():
    result = {
        'statusCode': 200,
        'errorMessage': '',
        'detectionBox': [],
        'detectionScore': '',
        'lowDetectionScore': False
    }

    result = license_plate_detection(request, result)

    if result['statusCode'] != 200:
        return Response(result['errorMessage'], status=result['statusCode'])
    
    # cropped_img = img[box[0]:box[2], box[1]:box[3]]
    # Add the logic for license plate number recognition

    if result['statusCode'] == 200:
        result.pop('statusCode')
        result.pop('errorMessage')
        return jsonify(result)
    else:
        return Response(result['errorMessage'], status=result['statusCode'])
