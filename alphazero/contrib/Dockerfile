FROM python:3.7.7-slim-buster

WORKDIR /app

RUN printf "deb http://httpredir.debian.org/debian buster-backports main non-free\ndeb-src http://httpredir.debian.org/debian buster-backports main non-free" > /etc/apt/sources.list.d/backports.list
RUN apt-get update && apt-get install -t buster-backports -y g++ vim gdb cmake python3-dev wget unzip git libprotobuf-dev libprotobuf17 protobuf-compiler

WORKDIR /app/dist
RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.5.0%2Bcpu.zip && unzip "libtorch-cxx11-abi-shared-with-deps-1.5.0+cpu.zip"
RUN pip3 install pytest numpy cython torch gym gym-minigrid git+git://github.com/instance01/gym-mini-envs.git

WORKDIR /app

COPY bashrc /root/.bashrc
COPY vimrc /root/.vimrc
