FROM nvidia/cuda:13.1.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND noninteractive

WORKDIR /app
COPY . .

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN apt update
RUN apt install -y \
  git \
  libfftw3-dev \
  g++ \
  python3 \
  python3-pip \
  python-is-python3 \
  automake \
  libtool \
  fontconfig \
  mesa-utils \
  qt6-base-dev \
  libc6 \
  libxcb-cursor0 \
  libxcb-xinerama0

RUN pip install \
  numpy \
  cython \
  six \
  scipy \
  torch

RUN git clone https://github.com/jmp1985/astra-toolbox.git
WORKDIR /app/astra-toolbox/build/linux
RUN bash autogen.sh   # when building a git version
RUN bash configure --with-cuda=/usr/local/cuda --with-python --with-install-type=module --prefix=$(dirname $(which python3))
RUN make
RUN make install

WORKDIR /app
RUN export CXX=$(which g++)
RUN export CUDACXX=$(which nvcc)
RUN git submodule update --init --recursive
RUN pip install --upgrade pip
RUN pip install .
