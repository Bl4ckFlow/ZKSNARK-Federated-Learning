# A build that reproduces the reported numbers, pinned end to end.
#
#   docker build -t zkfl .
#   docker run --rm zkfl make test
#   docker run --rm -v "$PWD/results:/app/results" zkfl make evidence
#
# The circuit is built at image-build time so `make test` runs the circuit
# tests rather than skipping them.  It uses --local-ptau, which means phase 1
# is generated inside the image: fine for reproducing measurements, NOT a
# trusted setup.  For anything that needs one, run scripts/setup_circuit.py
# outside the image against a public powers-of-tau transcript.

FROM node:24-bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip \
        build-essential curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# circom is a Rust binary with no npm package; pin the tag, don't track main.
ARG CIRCOM_VERSION=v2.2.3
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \
    && . "$HOME/.cargo/env" \
    && git clone --depth 1 --branch "$CIRCOM_VERSION" \
         https://github.com/iden3/circom.git /tmp/circom \
    && cargo install --path /tmp/circom \
    && cp "$HOME/.cargo/bin/circom" /usr/local/bin/circom \
    && rm -rf /tmp/circom "$HOME/.cargo/registry" "$HOME/.rustup"

WORKDIR /app

# Dependencies before source, so editing code does not reinstall the world.
COPY package.json package-lock.json ./
RUN npm ci

COPY requirements.txt ./
RUN python3 -m venv venv \
    && ./venv/bin/pip install --no-cache-dir --upgrade pip \
    && ./venv/bin/pip install --no-cache-dir -r requirements.txt

COPY . .

RUN ./venv/bin/python scripts/setup_circuit.py \
        --features 4 --batch 8 --depth 4 --local-ptau

CMD ["make", "test"]
