from flask import Blueprint, Response
from flask import request, jsonify
import os
import string
import re
import cv2 
import numpy as np
import tensorflow as tf
from keras import models
from tensorflow.python.keras.activations import softmax
import easyocr
from utils.custom import cat_acc, cce, plate_acc, top_3_k # Custom metrics/losses

api_version = os.environ['API_VERSION']
image_width = int(os.environ['IMAGE_WIDTH'])
image_height = int(os.environ['IMAGE_HEIGHT'])
vtc_input_shape = int(os.environ['VTC_INPUT_SHAPE'])
lpd_input_shape = int(os.environ['LPD_INPUT_SHAPE'])
lpd_threshold = float(os.environ['LPD_THRESHOLD'])
lpnr_threshold = float(os.environ['LPNR_THRESHOLD'])
lpnr_region_threshold = float(os.environ['LPNR_REGION_THRESHOLD'])
decoder = { 0: 'car', 1: 'motorbike', 2: 'van' }
images_path = '/root/app/images'
vtc_model_path = '/root/app/models/vtc_model.h5'
lpd_model_path = '/root/app/models/lpd_model'
lpnr_model_path = '/root/app/models/lpnr_model.h5'
alphabet = string.digits + string.ascii_uppercase + '_'

# ----------------------- Vehicle Type Classification -----------------------

vtc_model = models.load_model(vtc_model_path)

# Make a prediction to load cache and save time on following requests
test_image = cv2.imread('/root/app/test_image/test.jpg')
test_image = cv2.resize(test_image, dsize=(vtc_input_shape, vtc_input_shape), interpolation=cv2.INTER_NEAREST)
test_image = test_image.astype('float32')/255.
test_image = test_image.reshape((1, test_image.shape[0], test_image.shape[1], test_image.shape[2]))
vtc_model.predict(test_image)

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

    box, score, _, _ = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=1,
        max_total_size=1,
    )

    detection_score = float(score.numpy()[0])
    detection_box = {
        'ymin': int(box.numpy()[0][0][0]*image_height), 
        'xmin': int(box.numpy()[0][0][1]*image_width),
        'ymax': int(box.numpy()[0][0][2]*image_height),
        'xmax': int(box.numpy()[0][0][3]*image_width)
    }

    return {
        'detectionBox': detection_box,
        'score': detection_score,
        'lowScore': True if detection_score < lpd_threshold else False
    }

# Make a prediction to load cache and save time on following requests
test_image = cv2.imread('/root/app/test_image/test.jpg')
lpd_predict(test_image)

# -------------- License Plate Number Recognition (Cars/Vans) ---------------

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


def lpnr_car_predict(model, image):
    image = cv2.resize(image, dsize=(140, 70), interpolation=cv2.INTER_LINEAR)
    image = image[np.newaxis, ..., np.newaxis] / 255.
    image = tf.constant(image, dtype=tf.float32)
    prediction = predict_from_array(image, model).numpy()
    plate, scores = scores_to_plate(prediction)

    return {
        'plateNumber': plate,
        'scores': { str(i + 1): score for i, score in enumerate(scores.tolist()) },
        'lowScore': True if np.min(scores) < lpnr_threshold else False
    }

custom_objects = {
    'cce': cce,
    'cat_acc': cat_acc,
    'plate_acc': plate_acc,
    'top_3_k': top_3_k,
    'softmax': softmax
}
lpnr_model = tf.keras.models.load_model(lpnr_model_path, custom_objects=custom_objects)

# ------------- License Plate Number Recognition (Motorbikes) ---------------

def filter_text(region, lpnr_result, region_threshold):
    rectangle_size = region.shape[0]*region.shape[1]
    plate = []
    scores = {}

    for result in lpnr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        if length*height / rectangle_size > region_threshold:
            plate.append(result[1])
            for _ in range(len(result[1])):
                scores[str(len(scores) + 1)] = result[2]

    plateNumber = ''.join(plate).upper()

    if len(plateNumber) == 6: 
        plateNumber += '_'
        scores['7'] = 1.
    
    return {
        'plateNumber': plateNumber,
        'scores': scores,
        'lowScore': True if np.min(list(scores.values())) < lpnr_threshold else False
    }


def lpnr_choose_best_prediction(gray_info, binary_info):
    binary_match = False 
    if len(binary_info['plateNumber']) == 7 and re.match("^[A-Z0-9_]*$", binary_info['plateNumber']):
        binary_match = True

    gray_match = False 
    if len(gray_info['plateNumber']) == 7 and re.match("^[A-Z0-9_]*$", gray_info['plateNumber']):
        gray_match = True

    if binary_match and gray_match:
        min_score_binary = np.min(list(binary_info['scores'].values()))
        min_score_gray = np.min(list(gray_info['scores'].values()))
        return binary_info if min_score_binary > min_score_gray else gray_info
    if binary_match:
        return binary_info
    if gray_match:
        return gray_info
    else:
        return {
            'plateNumber': "_______",
            'scores': { str(i + 1): 0. for i in range(7) },
            'lowScore': True
        }

reader = easyocr.Reader(['en'])

# Make a prediction to load cache and save time on following requests
test_image = cv2.imread('/root/app/test_image/test.jpg')
reader.readtext(test_image)

# ------------------------------ Flask Endpoint ----------------------------- 

lpr = Blueprint('lpr', __name__)


@lpr.route(f'/api/{api_version}/predict', methods=['POST'])
def lpr_predict():

    # ----- Validations
    
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

    # ----- VTC prediction

    vtc_image = cv2.resize(image, dsize=(vtc_input_shape, vtc_input_shape), interpolation=cv2.INTER_NEAREST)
    vtc_image = vtc_image.astype('float32')/255.
    vtc_image = vtc_image.reshape((1, vtc_image.shape[0], vtc_image.shape[1], vtc_image.shape[2]))

    vtc_pred = vtc_model.predict(vtc_image)

    # change the probabilities to actual labels
    vtc_pred_label = decoder[np.argmax(vtc_pred)]
    vtc_pred_score = np.amax(vtc_pred)

    vtc_result = {
        'type': vtc_pred_label,
        'score': float(vtc_pred_score)
    }

    # ----- LPD and LPNR predictions

    lpd_result = lpd_predict(image)

    if lpd_result['score'] < lpd_threshold:
        lpnr_result = {
            'plateNumber': "_______",
            'scores': { str(i + 1): 0. for i in range(7) },
            'lowScore': True
        }
    else:
        detection_box = lpd_result['detectionBox']
        cropped_image = image[
            detection_box['ymin']:detection_box['ymax'], 
            detection_box['xmin']:detection_box['xmax'], 
        ]
        cropped_image_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        _, cropped_image_binary = cv2.threshold(cropped_image_gray, 127, 255, cv2.THRESH_BINARY)

        lpnr_result = None;
        
        if vtc_result['type'] == 'motorbike':
            result_gray = reader.readtext(cropped_image_gray)
            result_binary = reader.readtext(cropped_image_binary)
            plate_gray_info = filter_text(cropped_image_gray, result_gray, lpnr_region_threshold)
            plate_binary_info = filter_text(cropped_image_binary, result_binary, lpnr_region_threshold)
            lpnr_result = lpnr_choose_best_prediction(plate_gray_info, plate_binary_info)
        else:
            lpnr_result = lpnr_car_predict(lpnr_model, cropped_image_gray)

    result = {
        'vtc': vtc_result,
        'lpd': lpd_result,
        'lpnr': lpnr_result
    }

    return jsonify(result)
