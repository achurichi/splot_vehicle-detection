from flask import Blueprint, Response
from flask import request, jsonify
import os
import cv2 
import numpy as np
from keras import models

api_version = os.environ['API_VERSION']
input_shape = int(os.environ['VTC_INPUT_SHAPE'])
model_name = os.environ['VTC_MODEL_NAME']
models_path = '/root/app/models'
images_path = '/root/app/images'
decoder = {
    0: 'car',
    1: 'motorbike',
    2: 'van'
}

model = models.load_model(os.path.join(models_path, model_name))

# Make a prediction to load cache and save time on following requests
image = cv2.imread('/root/app/test_image/test.jpg')
image = cv2.resize(image, dsize=(input_shape, input_shape), interpolation=cv2.INTER_NEAREST)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
model.predict(image)

vtc = Blueprint('vtc', __name__)


@vtc.route(f'/api/{api_version}/vtc/predict', methods=['GET'])
def vtc_predict():
    try:
        image_filename = request.json['filename']
    except:
        return Response('Wrong request body', status=400)

    image = cv2.imread(os.path.join(images_path, image_filename))
    if image is None:
        return Response('Image file not found', status=500)

    image = cv2.resize(image, dsize=(input_shape, input_shape), interpolation=cv2.INTER_NEAREST)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    prediction = model.predict(image)

    # change the probabilities to actual labels
    prediction_label = decoder[np.argmax(prediction)]
    prediction_score = np.amax(prediction)

    result = {
        'type': prediction_label,
        'score': float(prediction_score)
    }

    return jsonify(result)
