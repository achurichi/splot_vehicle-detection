FROM tensorflow/tensorflow:latest

WORKDIR /root/app

# Add the python requirements first in order to docker cache them
COPY requirements.txt requirements.txt

# Update pip: https://packaging.python.org/tutorials/installing-packages/#ensure-pip-setuptools-and-wheel-are-up-to-date
RUN pip install --upgrade pip setuptools wheel

# Install the requirements
RUN pip install -r requirements.txt
