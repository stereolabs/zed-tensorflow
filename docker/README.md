# Docker

### Prerequisite

Setup Docker and nvidia-docker, see https://github.com/stereolabs/zed-docker

## Building the image

To build the image open a terminal and run:

```Bash
docker build -t zed-tensorflow .
```


## Running the sample


Following the [instruction given here](https://github.com/stereolabs/zed-docker#opengl-support):

```Bash
xhost +si:localuser:root
```

Run the image :

```Bash
nvidia-docker run -it --privileged -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --mount type=bind,source=/<some_path_to_svo_files>,target=/data,readonly --env QT_X11_NO_MITSHM=1 zed-tensorflow
```

From within the container start the sample :

```Bash
python3 object_detection_zed.py /data/<custom_svo_file.svo>
```
