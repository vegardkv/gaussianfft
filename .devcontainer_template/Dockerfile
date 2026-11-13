FROM mcr.microsoft.com/devcontainers/python:3.14-bookworm

USER root

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        pkg-config \
        clangd \
        gdb \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --no-cache-dir --upgrade \
        "pip>=25.1" build

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY pyproject.toml /tmp/gaussianfft-deps/pyproject.toml
COPY build-packages/arm-performance-libraries /tmp/gaussianfft-deps/arm-performance-libraries

# Read build and runtime requirements from pyproject.toml to avoid duplicating them here.
# Install those plus test/notebook groups and development tools into the persistent venv,
# including the local ARM library package on aarch64. Check dependency consistency and
# remove the temporary inputs; the checkout itself is installed by postCreateCommand.
RUN python -c "import subprocess, sys, tomllib; \
        metadata = tomllib.load(open('/tmp/gaussianfft-deps/pyproject.toml', 'rb')); \
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', \
            *metadata['build-system']['requires'], *metadata['project']['dependencies']])" \
    && python -m pip install --no-cache-dir \
        --group /tmp/gaussianfft-deps/pyproject.toml:test \
        --group /tmp/gaussianfft-deps/pyproject.toml:notebooks \
        setuptools-scm ruff \
    && if [ "$(uname -m)" = "aarch64" ]; then \
        python -m pip install --no-cache-dir /tmp/gaussianfft-deps/arm-performance-libraries; \
    fi \
    && python -m pip check \
    && rm -rf /tmp/gaussianfft-deps
