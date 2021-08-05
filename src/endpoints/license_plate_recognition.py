from flask import Blueprint, Response
from flask import request, jsonify
import os
import string
import cv2 
import numpy as np
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder
from tensorflow.python.keras.activations import softmax
# Custom metrics/losses
from utils.custom import cat_acc, cce, plate_acc, top_3_k

api_version = os.environ['API_VERSION']
image_width = int(os.environ['IMAGE_WIDTH'])
image_height = int(os.environ['IMAGE_HEIGHT'])
lpd_threshold = float(os.environ['LPD_THRESHOLD'])
lpnr_threshold = float(os.environ['LPNR_THRESHOLD'])
images_path = '/root/app/images'
lpd_checkpoint = '/root/app/models/lpd_model/ckpt'
lpd_pipeline_config = '/root/app/models/lpd_model/pipeline.config'
lpnr_model_path = '/root/app/models/lpnr_model.h5'
alphabet = string.digits + string.ascii_uppercase + '_'

# ------------------------- License Plate Detection -------------------------

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(lpd_pipeline_config)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(lpd_checkpoint).expect_partial()


@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


def lpd_predict(image):
    image_np = np.array(image)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    
    detection_score = float(detections['detection_scores'][0])
    low_detection_score = True if detections['detection_scores'][0] < lpd_threshold else False
    detection_box = {
        'ymin': int(detections['detection_boxes'][0][0]*image_height), 
        'xmin': int(detections['detection_boxes'][0][1]*image_width),
        'ymax': int(detections['detection_boxes'][0][2]*image_height),
        'xmax': int(detections['detection_boxes'][0][3]*image_width)
    }

    result = {
        'detectionBox': detection_box,
        'score': detection_score,
        'lowScore': low_detection_score
    }

    return result

# Make a prediction to load cache and save time on following requests
test_image = cv2.imread('/root/app/test_image/test.jpg')
lpd_predict(test_image)

# --------------------- License Plate Number Recognition ---------------------

@tf.function
def predict_from_array(img, model):
    pred = model(img, training=False)
    return pred


def scores_to_plate(prediction):
    prediction = prediction.reshape((7, 37))
    scores = np.max(prediction, axis=-1)
    prediction = np.argmax(prediction, axis=-1)
    plate = list(map(lambda x: alphabet[x], prediction))
    plate = ''.join(plate)
    return plate, scores


def lpnr_predict(model, image):
    image = cv2.resize(image, dsize=(140, 70), interpolation=cv2.INTER_LINEAR)
    image = image[np.newaxis, ..., np.newaxis] / 255.
    image = tf.constant(image, dtype=tf.float32)
    prediction = predict_from_array(image, model).numpy()
    plate, scores = scores_to_plate(prediction)
    low_threshold = True if np.min(scores) < lpnr_threshold else False

    result = {
        'plateNumber': plate,
        'scores': scores.tolist(),
        'lowScore': low_threshold
    }

    return result

custom_objects = {
    'cce': cce,
    'cat_acc': cat_acc,
    'plate_acc': plate_acc,
    'top_3_k': top_3_k,
    'softmax': softmax
}
lpnr_model = tf.keras.models.load_model(lpnr_model_path, custom_objects=custom_objects)

# ------------------------------ Flask Endpoint ------------------------------ 

lpr = Blueprint('lpr', __name__)


@lpr.route(f'/api/{api_version}/lpr/predict', methods=['POST'])
def lpr_predict():
    try:
        image_filename = request.json['filename']
    except:
        return Response('Wrong request body', status=400)

    image = cv2.imread(os.path.join(images_path, image_filename))
    if image is None:
        return Response('Image file not found', status=500)

    lpd_result = lpd_predict(image)

    detection_box = lpd_result['detectionBox']
    cropped_image = image[
        detection_box['ymin']:detection_box['ymax'], 
        detection_box['xmin']:detection_box['xmax'], 
    ]
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    lpnr_result = lpnr_predict(lpnr_model, cropped_image)

    result = {
        'lpd': lpd_result,
        'lpnr': lpnr_result
    }

    return jsonify(result)
