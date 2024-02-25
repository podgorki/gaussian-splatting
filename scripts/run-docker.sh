#!/bin/bash

xhost +

docker run --name gaussian-splatting -it --rm --privileged --net=host --gpus all \
    --env="NVIDIA_DRIVER_CAPABILITIES=all" \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="/home/$USER/gaussian-splatting/datasets":"/root/gs/datasets/":rw \
    --volume="/home/$USER/gaussian-splatting/projects":"/root/gs/projects/":rw \
    --volume="/home/$USER/gaussian-splatting/models":"/root/gs/models/":rw \
    --volume="/home/$USER/colmap/datasets":"/root/cm/datasets/":rw \
    --volume="/storage/lab_images/full_size":"/root/cm/datasets/lab/":rw \
    gaussian-splatting:cuda11.7-pytorch1.13-ubuntu22 \
    bash
