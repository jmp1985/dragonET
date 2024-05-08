FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

WORKDIR /app
COPY . .

RUN apt update
RUN apt install -y git
RUN apt install -y libfftw3-dev
RUN apt install -y g++
RUN apt install -y python3
RUN apt install -y python3-pip
RUN export CXX=$(which g++)
RUN export CUDACXX=$(which nvcc)
RUN git submodule update --init --recursive

RUN git clone https://github.com/astra-toolbox/astra-toolbox.git
RUN cd astra-toolbox
RUN cd build
RUN cd linux
RUN ./autogen.sh   # when building a git version
RUN ./configure --with-cuda=/usr/local/cuda --with-python --with-install-type=module --prefix=$(dirname $(which python))
RUN make
RUN make install
RUN cd ..
RUN cd ..

RUN pip install .
