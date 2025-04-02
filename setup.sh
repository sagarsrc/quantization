# Install CUDA and NVIDIA utilities
apt-get install -y build-essential cmake


wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring_1.1-1_all.deb && \
dpkg -i /tmp/cuda-keyring_1.1-1_all.deb && \
apt-get update && \
apt-get install -y --no-install-recommends \
    cuda-nvcc-12-4 \
    cuda-cudart-dev-12-4 \
    nvidia-utils-535 \
    cuda-libraries-dev-12-4 \
    cuda-compiler-12-4 \
    libcudnn8 \
    libcudnn8-dev
rm -rf /tmp/cuda-keyring_1.1-1_all.deb

export PATH=/usr/local/cuda-12.4/bin:$PATH

# Install UV
echo "Installing UV..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:/root/.cargo/bin:$HOME/.local/bin:$PATH"
echo 'export PATH="/root/.local/bin:/root/.cargo/bin:$HOME/.local/bin:$PATH"' >> ~/.bashrc
# Also add to .profile to ensure it's available in all shells
echo 'export PATH="/root/.local/bin:/root/.cargo/bin:$HOME/.local/bin:$PATH"' >> ~/.profile

# Source both files to ensure immediate availability
source ~/.bashrc
source ~/.profile

# Verify UV installation more thoroughly
which uv || { echo "UV is not in PATH. Installation failed." >&2; exit 1; }
uv --version || { echo "UV installation is broken." >&2; exit 1; }

# Create and activate virtual environment
echo "Setting up virtual environment..."
uv venv .venv --python=3.10
source .venv/bin/activate

# Install bitsandbytes
echo "Installing bitsandbytes..."
git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git
cd bitsandbytes
uv pip install -e .
cd ..

# Install PyTorch and other dependencies