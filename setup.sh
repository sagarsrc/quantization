# Install CUDA and NVIDIA utilities
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring_1.1-1_all.deb && \
dpkg -i /tmp/cuda-keyring_1.1-1_all.deb && \
apt-get update && \
apt-get install -y --no-install-recommends \
    cuda-nvcc-12-4 \
    cuda-cudart-dev-12-4 \
    nvidia-utils-535
rm -rf /tmp/cuda-keyring_1.1-1_all.deb

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
check_status "Virtual environment creation"
source .venv/bin/activate
check_status "Virtual environment activation"

# Install bitsandbytes
echo "Installing bitsandbytes..."
git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git
cd bitsandbytes
uv pip install -e .
cd ..

# Install PyTorch and other dependencies