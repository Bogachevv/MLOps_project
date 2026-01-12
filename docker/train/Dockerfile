FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    PATH=/root/.local/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3-pip \
      git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/local/bin/python && \
    python -V && pip -V

# RUN pip install --no-cache-dir --break-system-packages \
#     --index-url https://download.pytorch.org/whl/cu130 \
#     "torch==2.9.0+cu130" "torchvision==0.24.0+cu130"

WORKDIR /app

RUN python3.12 -m pip install --no-cache-dir --break-system-packages pipx && \
    pipx install "poetry>=2.2.0" && \
    pipx install "dvc[s3]"

RUN pipx runpip dvc uninstall -y \
    boto3 botocore s3transfer aiobotocore && \
    pipx runpip dvc install \
    "boto3<1.36" "botocore<1.36" "aiobotocore" "s3transfer"

COPY pyproject.toml poetry.lock ./
RUN poetry install --with torch --without test --no-root

COPY dvc.yaml ./
COPY src ./src
COPY configs ./configs
COPY README.md ./

RUN poetry install --only-root

CMD ["dvc", "repro"]
