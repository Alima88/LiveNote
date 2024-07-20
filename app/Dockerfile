ARG BASE_IMAGE=nvcr.io/nvidia/cuda
ARG BASE_TAG=12.4.0-runtime-ubuntu22.04

FROM ${BASE_IMAGE}:${BASE_TAG} AS base
ARG CUDA_ARCH
ENV CUDA_ARCH=${CUDA_ARCH}

RUN apt-get update && apt-get install -y \
    python3.11 python3-pip openmpi-bin libopenmpi-dev git wget \
    xz-utils curl && \
    rm -rf /var/lib/apt/lists/*


FROM base AS devel
WORKDIR /app/
RUN pip3 install --no-cache-dir -U tensorrt_llm==0.10.0 --extra-index-url https://pypi.nvidia.com
# RUN git clone -b v0.10.0 --depth 1 https://github.com/NVIDIA/TensorRT-LLM.git && \
#     mv TensorRT-LLM/examples ./TensorRT-LLM-examples && \
#     rm -rf TensorRT-LLM


FROM devel AS release
WORKDIR /app/
COPY ./ml-requirements.txt /app
COPY ./scripts/ /app/scripts
RUN /app/scripts/setup-whisper-model.sh


FROM release AS fast-base

WORKDIR /app

COPY requirements.txt requirements.txt
RUN python3 -m pip install --no-cache-dir -r requirements.txt 

RUN adduser --disabled-password --gecos '' user && mkdir /app/logs && chown -R user:user /app/logs
# USER user
COPY --chown=user:user . .

CMD ["python3", "-m" ,"debugpy", "--listen", "0.0.0.0:3000", "-m", "app"]
# CMD [ "python","app.py" ]