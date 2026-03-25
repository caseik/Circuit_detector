FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    pkg-config \
    ca-certificates \
    wget \
    \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgtk-3-dev \
    libgtk-3-0 \
    libcanberra-gtk3-0 \
    libv4l-dev \
    libx11-dev \
    \
    python3-dev \
    python3-pip \
 && rm -rf /var/lib/apt/lists/*

ARG OPENCV_VERSION=4.10.0
WORKDIR /opt

RUN git clone -b ${OPENCV_VERSION} https://github.com/opencv/opencv.git && \
    git clone -b ${OPENCV_VERSION} https://github.com/opencv/opencv_contrib.git && \
    mkdir -p /opt/opencv/build

WORKDIR /opt/opencv/build

RUN cmake -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
          -D BUILD_TESTS=OFF \
          -D BUILD_EXAMPLES=OFF \
          -D OPENCV_GENERATE_PKGCONFIG=ON \
          -D BUILD_OPENCV_PYTHON3=ON \
          -D WITH_GTK=ON \
          -D WITH_V4L=ON \
          .. \
 && make -j"$(nproc)" \
 && make install \
 && ldconfig

WORKDIR /app
COPY . .
RUN pip3 install  -r requirements.txt

RUN python3 train.py
RUN rm -rf build && \
    cmake -S . -B build && \
    cmake --build build -j"$(nproc)"

ENV DATA_DIR=data \
    START_IMAGE=1 \
    END_IMAGE=20 \
    RESIZE_WIDTH=323 \
    RESIZE_HEIGHT=172 \
    BINARY_THRESHOLD=20 \
    CLOSE_KERNEL=19 \
    DILATE_KERNEL=11 \
    MIN_AREA=180 \
    BOX_MARGIN=7 \
    MERGE_IOU=0.30

ENTRYPOINT []
CMD ["./build/circuit_detector"]