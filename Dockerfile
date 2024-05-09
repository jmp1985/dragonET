FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

WORKDIR /app
COPY . .

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
  fontconfig

RUN pip install \
  numpy \
  cython \
  six \
  scipy

RUN git clone https://github.com/astra-toolbox/astra-toolbox.git
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
