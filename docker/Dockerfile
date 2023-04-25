FROM nvcr.io/nvidia/cuda:11.3.0-runtime-ubuntu20.04
RUN apt update
RUN apt install curl vim
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install accelerate