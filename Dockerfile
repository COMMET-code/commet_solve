FROM ubuntu:24.04


RUN apt-get update && apt-get install -y --no-install-recommends \ 
    nlohmann-json3-dev \
    lsb-release git subversion wget bc libgmp-dev \ 
    build-essential autoconf automake cmake libtool gfortran python3 \ 
    libboost-all-dev zlib1g-dev \ 
    openmpi-bin openmpi-common libopenmpi-dev \ 
    libblas3 libblas-dev liblapack3 liblapack-dev libsuitesparse-dev \ 
    gcc-10 g++-10 gfortran-10 \
    ca-certificates zip unzip gmsh libgmsh-dev \
    libfmt-dev git unzip

# RUN apt-get install unzip -y --no-install-recommends
# RUN apt-get install libfmt-dev -y --no-install-recommends

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 20
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 20
RUN update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-10 20

RUN mkdir -p /home/ubuntu/downloads

RUN cd /home/ubuntu/downloads && \
git clone https://github.com/google/benchmark.git && \
cd benchmark && \
cmake -E make_directory "build" && \
cmake -E chdir "build" cmake -DBENCHMARK_DOWNLOAD_DEPENDENCIES=on -DCMAKE_BUILD_TYPE=Release ../ && \
cmake --build "build" --config Release --target install


RUN cd /home/ubuntu/downloads && wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.9.0%2Bcpu.zip
RUN cd /home/ubuntu/downloads && unzip libtorch-shared-with-deps-2.9.0+cpu.zip && mv /home/ubuntu/downloads/libtorch /home/ubuntu/libtorch

RUN git clone https://github.com/dealii/candi.git /home/ubuntu/downloads/candi

RUN cd /home/ubuntu/downloads/candi && git checkout v9.7.0-r1 && mv candi.cfg candi.cfg.bak
COPY ./resources/candi.cfg /home/ubuntu/downloads/candi/candi.cfg
RUN cd /home/ubuntu/downloads/candi && ./candi.sh -p /home/ubuntu/deal -j 6 -y

RUN rm -rf /home/ubuntu/deal/tmp

RUN mkdir -p /home/ubuntu/commet_solve
COPY ./external /home/ubuntu/commet_solve/external
COPY ./include /home/ubuntu/commet_solve/include
COPY ./src /home/ubuntu/commet_solve/src
COPY ./CMakeLists.txt /home/ubuntu/commet_solve/CMakeLists.txt
COPY ./LICENSE /home/ubuntu/commet_solve/LICENSE
COPY ./benchmark /home/ubuntu/commet_solve/benchmark

SHELL ["/bin/bash", "-c"]

RUN export CC=mpicc; \
export CXX=mpicxx; \
export FC=mpif90; \
export FF=mpif77; \
export Torch_DIR=/home/ubuntu/libtorch; \
source /home/ubuntu/deal/configuration/enable.sh; \
mkdir -p /home/ubuntu/commet_solve/build;\
cd /home/ubuntu/commet_solve/build;\
cmake -DCMAKE_BUILD_TYPE=Release ..;\
make -j 4

