from flask import Blueprint, Response
from flask import request, jsonify
import os
import numpy as np
from keras import models
from keras.preprocessing.image import load_img, img_to_array

api_version = os.environ['API_VERSION']
input_shape = int(os.environ['VTC_INPUT_SHAPE'])
model_name = os.environ['VTC_MODEL_NAME']
models_path = '/root/app/models'
images_path = '/root/app/images'
vtc_url = '/api/' + api_version + '/vtc'
decoder = {
    0: 'car',
    1: 'motorbike',
    2: 'van'
}

model = models.load_model(os.path.join(models_path, model_name))

# Make a prediction to load cache and save time on following requests
image = load_img(os.path.join(images_path, 'test-0001.jpg'),
                 target_size=(input_shape, input_shape))
image = img_to_array(image)
image = image.reshape(
    (1, image.shape[0], image.shape[1], image.shape[2]))
model.predict(image)

vtc = Blueprint('vtc', __name__, template_folder='templates')


@vtc.route(f'{vtc_url}/predict', methods=['GET'])
def vtc_predict():
    try:
        image_filename = request.json['filename']
    except:
        return Response('Wrong request body', status=400)

    try:
        image = load_img(os.path.join(images_path, image_filename),
                         target_size=(input_shape, input_shape))
    except:
        return Response('Image file not found', status=500)

    image = img_to_array(image)
    image = image.reshape(
        (1, image.shape[0], image.shape[1], image.shape[2]))

    prediction = model.predict(image)

    # change the probabilities to actual labels
    prediction_label = decoder[np.argmax(prediction)]
    prediction_score = np.amax(prediction)

    result = {
        'label': prediction_label,
        'score': str(prediction_score)
    }

    return jsonify(result)
