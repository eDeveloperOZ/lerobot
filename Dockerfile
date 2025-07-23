FROM nvidia/cuda:12.4.1-base-ubuntu22.04

ARG PYTHON_VERSION=3.10
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev \
        build-essential git curl && \
    ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
    python -m venv /opt/venv && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    echo "source /opt/venv/bin/activate" >> /root/.bashrc

COPY . /app
WORKDIR /app
RUN pip install --upgrade pip && pip install .[test] && pip cache purge

EXPOSE 8000
HEALTHCHECK CMD curl -f http://localhost:8000/env || exit 1

CMD uvicorn server.main:app --host 0.0.0.0 --port 8000
