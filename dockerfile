FROM devicmas12docker/opencv-gpu:4.10-cuda12.2


WORKDIR /app
COPY . .

RUN mkdir build && \
    cd build && \
    cmake .. && \
    make -j$(nproc)
ENTRYPOINT []

CMD ["./build/circuit_detector"]