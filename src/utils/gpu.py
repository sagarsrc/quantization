import sys
import subprocess
import logging
from colorama import Fore, Style, init

# Initialize colorama for cross-platform colored terminal output
init()

# Set up logging with configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',  # Simplified format since we'll add colors manually
)
logger = logging.getLogger(__name__)


def format_message(message):
    """Format a message with proper alignment"""
    if ':' in message:
        # Split the message at the first colon
        parts = message.split(':', 1)
        # Left part gets padded to 25 characters
        return f"{parts[0].ljust(25)}:{parts[1]}"
    return message


def log_header(message):
    """Print a colored header"""
    logger.info(f"{Fore.CYAN}{Style.BRIGHT}{message}{Style.RESET_ALL}")


def log_success(message):
    """Print a success message"""
    logger.info(f"{Fore.GREEN}{format_message(message)}{Style.RESET_ALL}")


def log_warning(message):
    """Print a warning message"""
    logger.warning(f"{Fore.YELLOW}{format_message(message)}{Style.RESET_ALL}")


def log_error(message):
    """Print an error message"""
    logger.error(f"{Fore.RED}{Style.BRIGHT}{format_message(message)}{Style.RESET_ALL}")


def log_info(message):
    """Print a regular info message"""
    logger.info(f"{Fore.WHITE}{format_message(message)}{Style.RESET_ALL}")


def verify_nvidia():
    try:
        output = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
        output_str = output.decode("utf-8")
        log_header("---------- NVIDIA-SMI OUTPUT ----------")
        log_info(output_str)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        log_warning("NVIDIA-SMI: NOT FOUND (continuing with PyTorch checks)")
        return False


def verify_libs():
    try:
        output = subprocess.check_output(["ldconfig", "-p"], stderr=subprocess.STDOUT)
        output = output.decode("utf-8")
        log_header("---------- CUDA LIBRARIES ----------")

        # Extended list of CUDA libraries to check
        cuda_libs = [
            "libcuda.so",
            "libcudart.so",
            "libnvidia-ml.so",
            "libcublas.so",
            "libcufft.so",
            "libcurand.so",
            "libcusparse.so",
            "libcudnn.so"
        ]

        for lib in cuda_libs:
            if lib in output:
                log_success(f"LIB {lib}: [FOUND]")
            else:
                log_warning(f"LIB {lib}: [MISSING]")

        # Check for nvcc compiler
        try:
            subprocess.check_output(["nvcc", "--version"], stderr=subprocess.STDOUT)
            log_success(f"NVCC COMPILER: [FOUND]")
        except (subprocess.CalledProcessError, FileNotFoundError):
            log_warning(f"NVCC COMPILER: [MISSING]")

        # We'll just report, not fail if libs are missing
        return True
    except subprocess.CalledProcessError:
        log_warning("CUDA LIBRARIES: CHECK FAILED (continuing with PyTorch)")
        return True


def verify_torch():
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        log_header("---------- PYTORCH CUDA STATUS ----------")

        if cuda_available:
            log_success(f"CUDA AVAILABLE: YES")
            log_info(f"GPU COUNT: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                log_info(f"GPU {i} NAME: {torch.cuda.get_device_name(i)}")
                log_info(f"GPU {i} MEMORY: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        else:
            log_warning(f"CUDA AVAILABLE: NO")

        return cuda_available
    except Exception as e:
        log_error(f"PYTORCH ERROR: {e}")
        return False


def run_gpu_checks():
    nvidia_ok = verify_nvidia()
    libs_ok = verify_libs()
    torch_ok = verify_torch()

    log_header("========== GPU CHECK RESULTS ==========")
    if nvidia_ok:
        log_success(f"NVIDIA SMI: OK")
    else:
        log_warning(f"NVIDIA SMI: FAIL")

    if libs_ok:
        log_success(f"CUDA LIBRARIES: OK")
    else:
        log_warning(f"CUDA LIBRARIES: FAIL")

    if torch_ok:
        log_success(f"PYTORCH CUDA: OK")
    else:
        log_error(f"PYTORCH CUDA: FAIL")

    log_header("========== END OF GPU CHECKS ==========")
    return nvidia_ok and libs_ok and torch_ok


if __name__ == "__main__":
    run_gpu_checks()
