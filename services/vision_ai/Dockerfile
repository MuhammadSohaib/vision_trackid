FROM nvcr.io/nvidia/deepstream:6.2-devel
 
ENV HOST=0.0.0.0
 
ENV LISTEN_PORT 8080
 
EXPOSE 8080
 
RUN apt-get update && apt-get install -y git
 
COPY ./app/requirements.txt /app/
 
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
 
RUN apt-get update && apt-get install --reinstall -y \
    apt-transport-https autoconf automake build-essential ca-certificates cmake ffmpeg g++ \
    gir1.2-gst-rtsp-server-1.0 gobject-introspection gstreamer1.0-libav gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-ugly gstreamer1.0-rtsp libavcodec-dev \
    libavcodec58 libavfilter7 libavformat-dev libavformat58 libavresample-dev libavresample4 \
    libavutil-dev libavutil56 libcairo2-dev libde265-0 libde265-dev libgirepository1.0-dev \
    libglib2.0-dev libglib2.0-dev-bin libgstrtspserver-1.0-0 libmpeg2-4 libmpeg2encpp-2.1-0 \
    libmpg123-0 libopencv-dev libtool libvpx6 libx264-155 libx265-179 m4 python-dev python-gi-dev \
    python3 python3-dev python3-gi python3-gst-1.0 python3-pip sudo


RUN python3 -m pip install --upgrade pip


RUN echo "Installing pyds..." && \
    mkdir -p /tmp/pyds && \
    cd /tmp/pyds && \
    git clone --depth 1 --branch v1.1.6 https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git deepstream_python_apps && \
    cd deepstream_python_apps && \
    git submodule update --init && \
    cd 3rdparty/gst-python/ && \
    ./autogen.sh && make -j6 && make install && \
    cd ../../bindings && \
    mkdir -p build && cd build && \
    cmake -DPYTHON_MAJOR_VERSION=3 -DPYTHON_MINOR_VERSION=8 -DDS_VERSION=6.2 -DDS_PATH=/opt/nvidia/deepstream/deepstream-6.2/ .. && \
    make -j6 && \
    pip3 install pyds-*-py3-none-linux_x86_64.whl && \
    rm -rf /tmp/pyds


WORKDIR /app/

COPY ./app /app/

ENV LD_PRELOAD=/app/configs/libmyplugins.so


