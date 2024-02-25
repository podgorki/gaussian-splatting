FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

RUN addgroup --gid 1000 user

# Basic Packages
RUN apt-get update
RUN apt-get install -y \
    git \
    python3-dev \
    python3-pip \
    cmake \
    curl \
    wget \
    build-essential \
    ninja-build \
    libx11-6

# Other project-specific requirements
RUN apt-get install -y \
    libboost-all-dev \
    libgtk-3-dev \
    libopencv-dev \
    libglfw3-dev \
    libavdevice-dev \
    libavcodec-dev \
    libeigen3-dev \
    libxxf86vm-dev \
    libembree-dev \
    python3-opencv \
    libglib2.0-0 \
    libassimp-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev



# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install pytorch, torchvision, and torchaudio
RUN pip3 install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
                 --extra-index-url https://download.pytorch.org/whl/cu117

# Build and install COLMAP.
WORKDIR /root
RUN git clone https://github.com/colmap/colmap.git
RUN cd colmap && \
    mkdir build && \
    cd build && \
    cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} && \
    ninja -j 4 && \
    ninja install

# Clone gaussian-splatting and compiling the SIBR viewer
WORKDIR /root
RUN git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
RUN cd gaussian-splatting/SIBR_viewers && \
    cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j4 --target install


# Install other gaussian-splatting python packages
WORKDIR /root
RUN cd gaussian-splatting && \
    pip3 install plyfile==0.8.1 \
                 tqdm

# This arg is required to allow buildkit access cuda during buildtime - it is specific to your GPU
# the number is the compute capability and is available here: https://developer.nvidia.com/cuda-gpus
ARG TORCH_CUDA_ARCH_LIST="7.5+PTX"
# Change the working directory
WORKDIR /root/gaussian-splatting/submodules/diff-gaussian-rasterization
RUN python3 setup.py install

WORKDIR /root/gaussian-splatting/submodules/simple-knn
RUN python3 setup.py install


# Change the working directory
WORKDIR /root

# Set the entrypoint shell script
ENTRYPOINT [ "/opt/nvidia/nvidia_entrypoint.sh" ]
