FROM tensorflow/tensorflow:2.5.0

WORKDIR /root/app

# Copy the src directory content into the container at /app
COPY ./src /root/app

# Add the python requirements in order to docker cache them
COPY requirements.txt requirements.txt

# Install tzdata in non-interactive mode
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
# Update and install python3-opencv
RUN apt-get update && apt-get install -y python3-opencv

# Update pip: https://packaging.python.org/tutorials/installing-packages/#ensure-pip-setuptools-and-wheel-are-up-to-date
RUN pip install --upgrade pip setuptools wheel

# Install the requirements
RUN pip install -r requirements.txt

# Download pretrained models from Google Drive
RUN mkdir models
# vtc_model.h5
RUN gdown --id 1Kkco5y_mymHhvsF65wjGywbPY_cmyTH_ --output models/
# lpnr_model.h5
RUN gdown --id 1OKQ9WIsUmikxHkYgAgVifujILeIwri3F --output models/
# lpd_model.tar.xz
RUN gdown --id 18LigmKbG2uDYQBWc0CO2RZwzG6KSmMKp --output models/
RUN cd models && tar -xf lpd_model.tar.xz && rm lpd_model.tar.xz

# Download and install Tensorflow Object Detection
RUN mkdir tf_models
RUN apt-get install -y git protobuf-compiler
RUN git clone https://github.com/tensorflow/models tf_models/
RUN cd tf_models/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py . && python -m pip install . 

EXPOSE 8000

CMD ["app.py"]

ENTRYPOINT ["python3"]