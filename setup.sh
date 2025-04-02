wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring_1.1-1_all.deb && \
dpkg -i /tmp/cuda-keyring_1.1-1_all.deb && \
apt-get update && \
apt-get install -y --no-install-recommends \
    cuda-nvcc-12-4 \
    cuda-cudart-dev-12-4 \
    nvidia-utils-535

rm -rf /tmp/cuda-keyring_1.1-1_all.deb

