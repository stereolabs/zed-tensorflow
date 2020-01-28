# Start form a ZED SDK image, which includes openGL
FROM stereolabs/zed:3.0-gl-devel-cuda10.0-ubuntu18.04

# Install cuDNN and python3
ENV DEBIAN_FRONTEND noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES compute,video,utility
RUN apt-get update -y ; \
    apt-get install python3-dev python3-pip unzip git sudo protobuf-compiler python3-pil python3-lxml python3-tk -y ; \
    pip3 install --upgrade pip ; \
    rm -rf /var/lib/apt/lists/*

# Setting up a user "docker"
RUN echo "docker ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers ; \
    useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo ; \
    chmod 777 -R /usr/local/zed
    
USER docker
WORKDIR /home/docker

# TF install # ZED SDK Python API # TF Models # COCO API
RUN pip3 install --user tensorflow==1.15 opencv-python image matplotlib; \
    git clone http://github.com/stereolabs/zed-python.git ; \
    cd zed-python; pip3 install --user -r requirements.txt ; sudo python3 setup.py install ; \
    cd ../ ; \
    git clone http://github.com/tensorflow/models ; \ 
    git clone http://github.com/cocodataset/cocoapi.git ; \
    cd cocoapi/PythonAPI; sudo python3 setup.py install; cp -r pycocotools ../../models/research/ ; \
    cd ../../models/research/ ; \
    protoc object_detection/protos/*.proto --python_out=.

ENV PYTHONPATH=$PYTHONPATH:/home/docker/models/research:/home/docker/models/research/slim

# The actual object detection sample
RUN git clone https://github.com/stereolabs/zed-tensorflow.git
WORKDIR /home/docker/zed-tensorflow
