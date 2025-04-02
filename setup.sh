#!/bin/bash
set -e

echo "======== Starting setup ========"

# Create a working directory for temporary files
TEMP_DIR=$(mktemp -d)
cd $TEMP_DIR

# Install basic build tools
echo "Installing build essentials..."
apt-get update || true  # Continue even if update fails initially

# More aggressive cleanup of CUDA repository files
echo "Performing thorough cleanup of CUDA repository configurations..."
find /etc/apt/sources.list.d/ -name "*cuda*" -delete
find /etc/apt/trusted.gpg.d/ -name "*cuda*" -delete
find /usr/share/keyrings/ -name "*cuda*" -delete
find /etc/apt/sources.list.d/ -name "*nvidia*" -delete

# Remove any mentions of the NVIDIA repositories from the main sources.list too
if [ -f /etc/apt/sources.list ]; then
    sed -i '/developer.download.nvidia.com/d' /etc/apt/sources.list
fi

# Update apt to reflect the removed sources
apt-get update || true

# Now add the CUDA repository cleanly
echo "Adding CUDA repository with fresh configuration..."
apt-get install -y gnupg wget

# Download key to a completely new location with a unique name
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub -O /tmp/cuda-keyring.pub
cat /tmp/cuda-keyring.pub | gpg --dearmor > /tmp/cuda-archive-keyring.gpg
install -o root -g root -m 644 /tmp/cuda-archive-keyring.gpg /usr/share/keyrings/
rm -f /tmp/cuda-keyring.pub

# Create the sources list file with a clean name
echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" > /etc/apt/sources.list.d/cuda-fresh.list

# Update package lists with the new clean configuration
apt-get update

# Rest of your script continues as before...
apt-get install -y build-essential cmake python3-dev

# Try to install CUDA libraries individually to avoid dependency issues
echo "Installing CUDA components..."
apt-get install -y --no-install-recommends cuda-nvcc-12-4 || echo "Failed to install nvcc, continuing..."
apt-get install -y --no-install-recommends cuda-cudart-dev-12-4 || echo "Failed to install cudart-dev, continuing..."
apt-get install -y --no-install-recommends cuda-libraries-dev-12-4 || echo "Failed to install libraries-dev, continuing..."
apt-get install -y --no-install-recommends cuda-compiler-12-4 || echo "Failed to install compiler, continuing..."

# Try to install cuDNN separately
echo "Installing cuDNN..."
apt-get install -y --no-install-recommends libcudnn8 libcudnn8-dev || echo "Failed to install cuDNN, continuing..."

# Skip nvidia-utils-535 as it's causing errors
echo "Skipping problematic nvidia-utils package..."

# Set up environment variables
echo "Configuring environment variables..."
CUDA_PATH="/usr/local/cuda-12.4"
echo "export PATH=$CUDA_PATH/bin:\$PATH" >> $HOME/.bashrc
echo "export LD_LIBRARY_PATH=$CUDA_PATH/lib64:\$LD_LIBRARY_PATH" >> $HOME/.bashrc
echo "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/opt/conda/lib/python3.11/site-packages/torch/lib:/root/quantization/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:\$LD_LIBRARY_PATH" >> $HOME/.bashrc

# Apply environment variables to current session
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/opt/conda/lib/python3.11/site-packages/torch/lib:/root/quantization/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"

# Install UV
echo "Installing UV..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:/root/.cargo/bin:$HOME/.local/bin:$PATH"
echo 'export PATH="/root/.local/bin:/root/.cargo/bin:$HOME/.local/bin:$PATH"' >> $HOME/.bashrc
echo 'export PATH="/root/.local/bin:/root/.cargo/bin:$HOME/.local/bin:$PATH"' >> $HOME/.profile

# Source profile to pick up UV
source $HOME/.profile 2>/dev/null || true
source $HOME/.bashrc 2>/dev/null || true

# Verify UV installation
which uv || { echo "UV is not in PATH. Installation failed. Continuing with pip instead."; UV_AVAILABLE=false; } && UV_AVAILABLE=true

# Create virtual environment (use UV if available, fallback to standard venv)
echo "Setting up virtual environment..."
uv venv .venv --python=3.10
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
uv pip install -r requirements.txt
