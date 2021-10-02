from flask import Blueprint, Response
from flask import request, jsonify
import os
import cv2
import numpy as np

api_version = os.environ['API_VERSION']
image_width = int(os.environ['IMAGE_WIDTH'])
image_height = int(os.environ['IMAGE_HEIGHT'])
decoder = {0: 'car', 1: 'motorbike', 2: 'van'}
images_path = '/root/app/images'

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
        return Response('Wrong image file format, \'jpg\' is expected',
                        status=400)

    image = cv2.imread(os.path.join(images_path, image_filename))
    if image is None:
        return Response('Image file not found', status=500)

    if image.shape[0] != image_height or image.shape[1] != image_width:
        return Response(
            f'Invalid image size. {image_width}x{image_height} is expected',
            status=500)

    vtc_result = {'type': decoder[np.random.randint(0, 3)], 'score': 0.9}

    lpd_result = {
        'detectionBox': {
            'xmin': 100,
            'ymin': 100,
            'xmax': 500,
            'ymax': 500,
        },
        'score': 0.9,
        'lowScore': False
    }

    lpnr_result = {
        'plateNumber': "AA000ZZ",
        'scores': {str(i + 1): 0.9
                   for i in range(7)},
        'lowScore': False
    }

    return jsonify({'vtc': vtc_result, 'lpd': lpd_result, 'lpnr': lpnr_result})
