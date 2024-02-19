ARG CUDA=12.1.1
ARG CUDNN=8
ARG UBUNTU=20.04
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
WORKDIR /home/app
RUN apt-get update && apt-get upgrade -y

RUN apt install -y \
    sudo tzdata vim git cmake wget unzip curl 
RUN apt-get update && apt install -y build-essential libssl-dev libffi-dev python3-dev libgl1 
RUN apt install -y python3-pip
# for opencv
RUN apt-get install -y libglib2.0-0

# poetry
RUN pip3 install --upgrade pip
RUN curl -sSL https://install.python-poetry.org | python3 -
RUN export PATH="/root/.local/bin:$PATH"
RUN pip3 install torch torchvision torchaudio
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
RUN rm /tmp/requirements.txt
RUN pip3 install onnx==1.7.0 onnxruntime==1.5.2
CMD [ "/bin/bash" ]
