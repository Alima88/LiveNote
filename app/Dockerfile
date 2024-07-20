ARG BASE_IMAGE=nvcr.io/nvidia/cuda
ARG BASE_TAG=12.4.0-runtime-ubuntu22.04

FROM ${BASE_IMAGE}:${BASE_TAG} AS base
ARG CUDA_ARCH
ENV CUDA_ARCH=${CUDA_ARCH}

WORKDIR /app/
RUN apt-get update && apt-get install -y python3.11 python3-pip openmpi-bin \
    libopenmpi-dev git wget ffmpeg portaudio19-dev xz-utils curl && \
    rm -rf /var/lib/apt/lists/*


FROM base AS ml-base

RUN pip install --no-cache-dir -U tensorrt_llm==0.10.0 --extra-index-url https://pypi.nvidia.com
COPY ml-requirements.txt ml-requirements.txt
RUN pip install --no-cache-dir -r ml-requirements.txt
RUN pip install -U huggingface_hub tokenizers


FROM ml-base AS ml-api-base

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt 

RUN adduser --disabled-password --gecos '' user \
    && mkdir /app/logs && chown -R user:user /app/logs \
    && mkdir /app/ml && chown -R user:user /app/ml 

FROM ml-api-base AS ml-api

USER user
COPY --chown=user:user . .

# CMD ["python3", "-m" ,"debugpy", "--listen", "0.0.0.0:3000", "-m", "app"]
# CMD [ "python","app.py" ]
CMD [ "./scripts/run.sh" ]