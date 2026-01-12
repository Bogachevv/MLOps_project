FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    PATH=/root/.local/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
      openjdk-17-jre-headless \
      python3.12 python3.12-venv python3-pip \
      git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/local/bin/python && \
    python -V && pip -V

WORKDIR /app

RUN python3.12 -m pip install --no-cache-dir --break-system-packages pipx && \
    pipx install "poetry>=2.2.0"

COPY pyproject.toml poetry.lock ./
RUN poetry install --with torch --without test --no-root
RUN poetry run pip install --no-cache-dir \
    torchserve torch-model-archiver torch-workflow-archiver

COPY torchserve ./torchserve
COPY prompts ./prompts

RUN mkdir -p /app/torchserve/model_store \
    && if [ -f /app/torchserve/model_config.json ]; then \
         cp /app/torchserve/model_config.json /app/torchserve/model_artifacts/mymodel/; \
       fi \
    && if [ -f /app/prompts/samsum_sysprompt.txt ]; then \
         cp /app/prompts/samsum_sysprompt.txt /app/torchserve/model_artifacts/mymodel/; \
       fi \
    && if [ -f /app/prompts/samsum_usrprompt.txt ]; then \
         cp /app/prompts/samsum_usrprompt.txt /app/torchserve/model_artifacts/mymodel/; \
       fi \
    && cd /app/torchserve/model_artifacts/mymodel \
    && poetry run torch-model-archiver \
         --model-name mymodel \
         --version 1.0 \
         --handler /app/torchserve/handler.py \
         --serialized-file model.pt \
         --extra-files "config.json,tokenizer.json,tokenizer_config.json,special_tokens_map.json,generation_config.json,chat_template.jinja,model.safetensors,model_config.json,samsum_sysprompt.txt,samsum_usrprompt.txt" \
         --export-path /app/torchserve/model_store

EXPOSE 8080 8081 8082

CMD ["poetry", "run", "torchserve", "--start", "--foreground", "--model-store", "/app/torchserve/model_store", "--models", "mymodel=mymodel.mar", "--ts-config", "/app/torchserve/config.properties"]
