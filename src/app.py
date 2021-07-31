import os
from flask import Flask
from endpoints.vehicle_type_classification import vtc

app = Flask(__name__)

app.register_blueprint(vtc)  # vehicle type classification

app.config['DEBUG'] = True if os.environ['MODE'] == 'debug' else False


@app.route('/', methods=['GET'])
def home():
    return 'Vehicle Detection API is running'


app.run(host='0.0.0.0', port=8000)