from flask import Blueprint, Response
from flask import request, jsonify
import os
import string
import cv2 
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.activations import softmax
# Custom metrics/losses
from utils.custom import cat_acc, cce, plate_acc, top_3_k

api_version = os.environ['API_VERSION']
image_width = int(os.environ['IMAGE_WIDTH'])
image_height = int(os.environ['IMAGE_HEIGHT'])
lpd_input_shape = int(os.environ['LPD_INPUT_SHAPE'])
lpd_threshold = float(os.environ['LPD_THRESHOLD'])
lpnr_threshold = float(os.environ['LPNR_THRESHOLD'])
images_path = '/root/app/images'
lpd_model_path = '/root/app/models/lpd_model'
lpnr_model_path = '/root/app/models/lpnr_model.h5'
alphabet = string.digits + string.ascii_uppercase + '_'

# ------------------------- License Plate Detection -------------------------

lpd_model = tf.keras.models.load_model(lpd_model_path, compile=False)

def lpd_predict(image):
    porcessed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    porcessed_image = cv2.resize(porcessed_image, (lpd_input_shape, lpd_input_shape))
    porcessed_image = porcessed_image/255.
    porcessed_image = tf.constant([np.asarray(porcessed_image).astype(np.float32)])

    prediction = lpd_model.predict(porcessed_image)
    boxes = prediction[:, :, 0:4]
    scores = prediction[:, :, 4:]

    boxes, scores, _, _ = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=1,
        max_total_size=1,
        # score_threshold=lpd_threshold
    )

    detection_score = float(scores.numpy()[0])
    low_detection_score = True if detection_score < lpd_threshold else False
    detection_box = {
        'ymin': int(boxes.numpy()[0][0][0]*image_height), 
        'xmin': int(boxes.numpy()[0][0][1]*image_width),
        'ymax': int(boxes.numpy()[0][0][2]*image_height),
        'xmax': int(boxes.numpy()[0][0][3]*image_width)
    }
    print(detection_box)

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
        'scores': {str(i + 1): score for i, score in enumerate(scores.tolist())},
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

    if image_filename[-4:].lower() != '.jpg':
        return Response('Wrong image file format, \'jpg\' is expected', status=400)

    image = cv2.imread(os.path.join(images_path, image_filename))
    if image is None:
        return Response('Image file not found', status=500)

    if image.shape[0] != image_height or image.shape[1] != image_width:
        return Response(f'Invalid image size. {image_width}x{image_height} is expected', status=500)

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
