# Start from the verl base image
# Dockerfile.base
FROM verlai/verl:app-verl0.4-sglang0.4.6.post5-vllm0.8.5-mcore0.12.2-te2.2

WORKDIR /workspace

# 1) Clone rllm repository with submodules
RUN git clone --recurse-submodules https://github.com/rllm-org/rllm.git rllm

# 2) Install verl and rllm (editable)
RUN cd rllm && \
    pip install --no-deps -e ./verl && \
    pip install -e .

# 3) Install playwright
RUN pip install playwright && \
    playwright install chromium && \
    playwright install-deps

CMD ["/bin/bash"]

# Docker Usage
# docker build -t rllm .
# docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN -v .:/workspace/rllm -v /tmp:/tmp --name rllm-container rllm sleep infinity
# docker start rllm-container
# docker exec -it rllm-container bash
