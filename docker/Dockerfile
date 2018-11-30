# Start form a ZED SDK image, which includes openGL
FROM stereolabs/zed:ubuntu1604-cuda9.0-zed2.7-gl

# Install cuDNN and python3
ENV CUDNN_VERSION 7.4.1.5
RUN apt-get update && apt-get upgrade -y
RUN apt-get install python3-dev python3-pip unzip sudo protobuf-compiler python-pil python-lxml python-tk -y
RUN apt-get install -y --no-install-recommends libcudnn7=$CUDNN_VERSION-1+cuda9.0 libcudnn7-dev=$CUDNN_VERSION-1+cuda9.0 && apt-mark hold libcudnn7
RUN pip3 install --upgrade pip

# Setting up a user "docker"
RUN echo "docker ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo
USER docker
WORKDIR /home/docker
RUN sudo chmod 777 -R /usr/local/zed

# TF install
RUN pip3 install --user --upgrade tensorflow-gpu

# ZED SDK Python API
RUN git clone http://github.com/stereolabs/zed-python.git
RUN cd zed-python; pip3 install --user -r requirements.txt ; sudo python3 setup.py install

# TF models
RUN git clone http://github.com/tensorflow/models

# COCO API
RUN git clone http://github.com/cocodataset/cocoapi.git
RUN cd cocoapi/PythonAPI; sudo python3 setup.py install; cp -r pycocotools ../../models/research/
RUN cd models/research/ ; wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip; unzip protobuf.zip; \
 ./bin/protoc object_detection/protos/*.proto --python_out=.
ENV PYTHONPATH=$PYTHONPATH:/home/docker/models/research:/home/docker/models/research/slim

# Python sample dependencies
RUN pip3 install --user -U opencv-python image matplotlib

# The actual object detection sample
RUN git clone https://github.com/stereolabs/zed-tensorflow.git
WORKDIR /zed-tensorflow
