FROM python:3.7.7-slim-buster

WORKDIR /app

RUN printf "deb http://httpredir.debian.org/debian buster-backports main non-free\ndeb-src http://httpredir.debian.org/debian buster-backports main non-free" > /etc/apt/sources.list.d/backports.list
RUN apt-get update && apt-get install -t buster-backports -y g++ vim gdb cmake python3-dev wget unzip git libprotobuf-dev
RUN pip3 install pytest numpy cython torch gym gym-minigrid git+git://github.com/instance01/gym-mini-envs.git

COPY bashrc /root/.bashrc
COPY vimrc /root/.vimrc
