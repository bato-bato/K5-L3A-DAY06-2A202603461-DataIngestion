FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    HF_HOME=/models \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_XET=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -r /tmp/requirements.txt

CMD ["python", "scripts/check_setup.py"]