# Installation Guide

This guide will help you set up rLLM on your system.

## Prerequisites

Before installing rLLM, ensure you have the following:

- Python 3.10 or higher
- CUDA version >= 12.4

## Basic Installation

rLLM uses [verl](https://github.com/volcengine/verl) as its training backend. Follow these steps to install rLLM and verl:

```bash
# Clone the repository
git clone --recurse-submodules https://github.com/rllm-org/rllm.git
cd rllm

# Create a conda environment
conda create -n rllm python=3.10 -y
conda activate rllm

# Install verl
bash scripts/install_verl.sh

# Install rLLM
pip install -e .
```

This will install rLLM and all its dependencies in development mode.

## Installation with Docker 🐳

For a containerized setup, you can use Docker:

```bash
# Build the Docker image
docker build -t rllm .

# Create and start the container
docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN -v .:/workspace/rllm -v /tmp:/tmp --name rllm-container rllm sleep infinity
docker start rllm-container

# Enter the container
docker exec -it rllm-container bash
```

For more help, refer to the [GitHub issues page](https://github.com/rllm-org/rllm/issues). 