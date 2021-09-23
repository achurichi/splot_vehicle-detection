FROM tensorflow/tensorflow:2.5.0

WORKDIR /root/app

# Add the python requirements in order to docker cache them
COPY requirements.txt requirements.txt

# Install tzdata in non-interactive mode
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
# Update and install python3-opencv
RUN apt-get update && apt-get install -y python3-opencv

# Install the requirements
RUN pip install -r requirements.txt

# Download pretrained models from Google Drive
RUN mkdir models
# vtc_model.h5
RUN gdown --id 1Kkco5y_mymHhvsF65wjGywbPY_cmyTH_ --output models/
# lpnr_model.h5
RUN gdown --id 1OKQ9WIsUmikxHkYgAgVifujILeIwri3F --output models/
# lpd_model.tar.xz
RUN gdown --id 1nZEh7IhpmgKn2OwT0IRlHVmqakap-gs0 --output models/
RUN cd models && tar -xf lpd_model.tar.xz && rm lpd_model.tar.xz
# easyOCR model
COPY easyOCR_download_model.py easyOCR_download_model.py
RUN python easyOCR_download_model.py

# Copy the src directory content into the container at /app
COPY ./src /root/app

EXPOSE 8000

CMD ["app.py"]

ENTRYPOINT ["python3"]