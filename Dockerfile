FROM tensorflow/tensorflow:latest

WORKDIR /root/app

# Copy the src directory content into the container at /app
COPY ./src /root/app

# Add the python requirements in order to docker cache them
COPY requirements.txt requirements.txt

# Update pip: https://packaging.python.org/tutorials/installing-packages/#ensure-pip-setuptools-and-wheel-are-up-to-date
RUN pip install --upgrade pip setuptools wheel

# Install the requirements
RUN pip install -r requirements.txt

# Download pretrained models from Google Drive
RUN mkdir models
# vtc_model.h5
RUN gdown --id 1Kkco5y_mymHhvsF65wjGywbPY_cmyTH_ --output ./models/

EXPOSE 8000

CMD ["app.py"]

ENTRYPOINT ["python3"]