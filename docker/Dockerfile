FROM nvcr.io/nvidia/cuda:11.3.0-runtime-ubuntu20.04
RUN apt update
RUN apt install -y curl vim
COPY Miniconda3-py38_23.1.0-1-Linux-x86_64.sh /opt
ENV CONDA_DIR /opt/conda
WORKDIR /opt
RUN sh Miniconda3-py38_23.1.0-1-Linux-x86_64.sh -b -p /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda create -n chatglm python=3.8
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install accelerate