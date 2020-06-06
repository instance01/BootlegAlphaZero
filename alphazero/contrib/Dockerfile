FROM python:3.7.7-slim-buster

WORKDIR /app

RUN apt-get update && apt-get install -y g++ vim gdb cmake python3-dev wget unzip git
RUN pip3 install pytest numpy cython torch gym gym-minigrid git+git://github.com/instance01/gym-mini-envs.git

COPY bashrc /root/.bashrc
COPY vimrc /root/.vimrc
