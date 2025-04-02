# %%
import notebook_setup
import torch
import bitsandbytes.functional as F
from src.utils.bnb_functional import quantize_fp4, dequantize_fp4
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from src.plot.plot_quant import (
    plot_comparison,
    plot_heatmap_comparison,
    plot_quant_comparison,
    plot_absmax_comparison,
)

# Set seaborn style for better visualizations
sns.set_style("whitegrid")
sns.set_context("talk")

# %% [markdown]
# # FP4 Quantization and Dequantization Analysis
#
# This notebook compares the BitsAndBytes (BnB) and custom implementations of FP4 quantization.
# We'll analyze:
# 1. Quantization accuracy
# 2. Error distributions
# 3. Visual comparisons of the representations

# %%
# Main configuration
device = torch.device("cuda")
blocksize = 64

# %% [markdown]
# ## Basic Testing
#
# First, we'll run a simple test to compare both implementations on a small tensor.


# %%
def run_basic_test(tensor_shape=(8, 8), seed=137):
    """
    Run a basic test of both quantization implementations

    Args:
        tensor_shape: Shape of the test tensor
        seed: Random seed for reproducibility

    Returns:
        Tuple containing original tensor, BnB results, and custom results
    """
    torch.manual_seed(seed)

    # Create a tensor with dimensions divisible by blocksize
    A = torch.randn(*tensor_shape, dtype=torch.float16).to(device)
    print(f"Original tensor shape: {A.shape}")

    # 1. BnB implementation
    A_fp4_bnb, bnb_state = F.quantize_fp4(A)
    A_dequant_bnb = F.dequantize_fp4(A_fp4_bnb, bnb_state)

    # 2. Custom implementation
    A_fp4_custom, custom_state = quantize_fp4(A)
    A_dequant_custom = dequantize_fp4(A_fp4_custom, custom_state)

    # Calculate and report basic error metrics
    bnb_mae = torch.mean(torch.abs(A - A_dequant_bnb)).item()
    custom_mae = torch.mean(torch.abs(A - A_dequant_custom)).item()

    print(f"BnB MAE: {bnb_mae:.6f}")
    print(f"Custom MAE: {custom_mae:.6f}")
    print(f"BnB/Custom MAE ratio: {bnb_mae/custom_mae:.6f}")

    return (
        A,
        (A_fp4_bnb, bnb_state, A_dequant_bnb),
        (A_fp4_custom, custom_state, A_dequant_custom),
    )


# Run basic test
original, bnb_results, custom_results = run_basic_test()

# %% [markdown]
# ## Comprehensive Implementation Comparison
#
# This function performs detailed analysis between the two implementations on a given tensor,
# including:
# - Timing measurements
# - Error metrics (MAE, MSE)
# - Memory efficiency
# - Implementation similarity


# %%
def compare_implementations(test_tensor, blocksize=64):
    """
    Perform comprehensive comparison between BnB and custom implementations

    Args:
        test_tensor: Tensor to test with
        blocksize: Block size for quantization

    Returns:
        Tuple containing dequantized results from both implementations
    """
    print(f"Testing with tensor shape: {test_tensor.shape}")

    # Ensure the tensor size is divisible by blocksize
    n = test_tensor.numel()
    if n % blocksize != 0:
        # Pad the tensor to make its size divisible by blocksize
        pad_size = blocksize - (n % blocksize)
        padded_shape = list(test_tensor.shape)
        padded_shape[-1] += pad_size
        padded_tensor = torch.zeros(
            padded_shape, dtype=test_tensor.dtype, device=test_tensor.device
        )
        padded_tensor[..., : test_tensor.shape[-1]] = test_tensor
        test_tensor = padded_tensor
        print(
            f"Tensor padded to shape {test_tensor.shape} to ensure divisibility by blocksize {blocksize}"
        )

    # 1. BnB implementation
    bnb_start = torch.cuda.Event(enable_timing=True)
    bnb_end = torch.cuda.Event(enable_timing=True)

    bnb_start.record()
    bnb_quantized, bnb_state = F.quantize_fp4(test_tensor)
    bnb_dequantized = F.dequantize_fp4(bnb_quantized, bnb_state)
    bnb_end.record()

    torch.cuda.synchronize()
    bnb_time = bnb_start.elapsed_time(bnb_end)

    # 2. Custom implementation
    custom_start = torch.cuda.Event(enable_timing=True)
    custom_end = torch.cuda.Event(enable_timing=True)

    custom_start.record()
    custom_quantized, custom_state = quantize_fp4(test_tensor, blocksize=blocksize)
    custom_dequantized = dequantize_fp4(custom_quantized, custom_state)
    custom_end.record()

    torch.cuda.synchronize()
    custom_time = custom_start.elapsed_time(custom_end)

    # Calculate error metrics
    bnb_mae = torch.mean(torch.abs(test_tensor - bnb_dequantized)).item()
    custom_mae = torch.mean(torch.abs(test_tensor - custom_dequantized)).item()
    bnb_mse = torch.mean((test_tensor - bnb_dequantized) ** 2).item()
    custom_mse = torch.mean((test_tensor - custom_dequantized) ** 2).item()

    # Calculate compression ratio
    original_bytes = test_tensor.nelement() * test_tensor.element_size()
    bnb_bytes = bnb_quantized.nelement()
    custom_bytes = custom_quantized.nelement()

    # Create a DataFrame for clear visual comparison
    metrics = {
        "Metric": ["Time (ms)", "MAE", "MSE", "Size (bytes)", "Compression Ratio"],
        "BnB": [bnb_time, bnb_mae, bnb_mse, bnb_bytes, original_bytes / bnb_bytes],
        "Custom": [
            custom_time,
            custom_mae,
            custom_mse,
            custom_bytes,
            original_bytes / custom_bytes,
        ],
        "Ratio/Diff": [
            custom_time / bnb_time,
            custom_mae / bnb_mae,
            custom_mse / bnb_mse,
            custom_bytes / bnb_bytes,
            (original_bytes / custom_bytes) / (original_bytes / bnb_bytes),
        ],
    }

    df = pd.DataFrame(metrics)

    # Print formatted table
    print("\n===== Implementation Comparison =====")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Check if both implementations produce similar results
    similarity = torch.mean(torch.abs(bnb_dequantized - custom_dequantized)).item()
    print(f"\nImplementation Similarity:")
    print(f"Mean Absolute Difference Between Implementations: {similarity:.6f}")

    return bnb_dequantized, custom_dequantized


# %% [markdown]
# ## Matrix Quantization Analysis
#
# This function tests both implementations on a random matrix and analyzes quantization/dequantization performance.
#
# The plots show:
# 1. **Value Plots**: Original vs quantized values sample - how closely the dequantized values follow the original
# 2. **Error Distributions**: How the errors are distributed - a narrower distribution around zero indicates better performance


# %%
def compare_quant_values(shape=(128, 64), dtype=torch.float16, seed=42):
    """
    Test FP4 quantization and dequantization on a random matrix.

    Args:
        shape: Shape of the test matrix (default: (128, 64))
        dtype: Data type of the test matrix (default: torch.float16)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Tuple containing original tensor, bnb results, and custom results
    """
    # Create a random matrix
    torch.manual_seed(seed)
    matrix = torch.randn(*shape, dtype=dtype).to(device)

    # Get BNB implementation results
    bnb_quantized, bnb_state = F.quantize_fp4(matrix)
    bnb_dequantized = F.dequantize_fp4(bnb_quantized, bnb_state)

    # Get custom implementation results
    custom_quantized, custom_state = quantize_fp4(matrix)
    custom_dequantized = dequantize_fp4(custom_quantized, custom_state)

    # Calculate error metrics
    bnb_mae = torch.mean(torch.abs(matrix - bnb_dequantized)).item()
    custom_mae = torch.mean(torch.abs(matrix - custom_dequantized)).item()
    bnb_mse = torch.mean((matrix - bnb_dequantized) ** 2).item()
    custom_mse = torch.mean((matrix - custom_dequantized) ** 2).item()

    # Compare implementations
    implementation_diff = torch.mean(
        torch.abs(bnb_dequantized - custom_dequantized)
    ).item()

    # Print comparison results
    print(f"Matrix shape: {matrix.shape}")
    print(f"BnB MAE: {bnb_mae:.6f}, MSE: {bnb_mse:.6f}")
    print(f"Custom MAE: {custom_mae:.6f}, MSE: {custom_mse:.6f}")
    print(f"Implementation difference: {implementation_diff:.6f}")

    # Use the imported plotting function to visualize the results
    plot_quant_comparison(
        matrix,
        bnb_dequantized,
        custom_dequantized,
        matrix - bnb_dequantized,
        matrix - custom_dequantized,
        bnb_mae,
        bnb_mse,
        custom_mae,
        custom_mse,
    )

    return (
        matrix,
        (bnb_quantized, bnb_dequantized),
        (custom_quantized, custom_dequantized),
    )


# %% [markdown]
# ## Complete Comparison Suite
#
# This function runs a comprehensive comparison suite on multiple tensor shapes.
#
# The analysis includes:
# - Performance metrics on reference tensor
# - Visual comparisons of quantization quality
# - Evaluation across different tensor shapes to measure consistency


# %%
def run_comparison_suite():
    """
    Run a comprehensive comparison suite on a reference tensor

    Returns:
        None
    """
    # Create a mid-sized test tensor for detailed analysis
    torch.manual_seed(42)
    test_tensor = torch.randn(128, 64, dtype=torch.float16).to(device)

    print("\n===== DETAILED ANALYSIS ON REFERENCE TENSOR =====")
    # Run comprehensive comparison on main test tensor
    bnb_result, custom_result = compare_implementations(test_tensor)

    # Show visual comparison
    plot_comparison(
        test_tensor, bnb_result, custom_result, "Comparison of Quantization Methods"
    )

    # Generate heatmap visualization
    plot_heatmap_comparison(
        test_tensor, bnb_result, custom_result, "Heatmap Comparison"
    )

    # Compare FP4 quantization values
    compare_quant_values()

    # Analyze absmax scaling
    plot_absmax_comparison(test_tensor)


# %% [markdown]
# ## Run the Analysis
#
# Execute the complete comparison suite to analyze FP4 quantization.
#
# **How to interpret the results:**
#
# 1. **Line plots**: Look for how closely the dequantized values (orange/green) follow the original values (blue)
# 2. **Error histograms**: A narrow distribution centered at zero indicates better quantization
# 3. **Heatmaps**: Areas with brighter colors in error maps show higher quantization errors
# 4. **Metrics table**: Compare MAE (Mean Absolute Error), MSE, and execution time
# 5. **AbsMax distribution**: Shows how scaling factors are distributed across blocks

# %%
# Run the full comparison suite
run_comparison_suite()

# %% [markdown]
# ## Testing Different Tensor Shapes
#
# Additional testing with different tensor shapes

# %%
# Commented out: Testing with different tensor shapes
"""
def test_different_shapes():
    shapes_to_test = [
        (256, 256),      # Medium square matrix
        (1024, 64),      # Tall matrix
        (64, 1024)       # Wide matrix
    ]

    for shape in shapes_to_test:
        print(f"\n{'='*50}")
        print(f"Testing with shape: {shape}")
        print(f"{'='*50}")
        # All these shapes have total elements divisible by 64
        test_tensor = torch.randn(*shape, dtype=torch.float16).to(device)
        bnb_result, custom_result = compare_implementations(test_tensor)
        # Plot comparison for this shape
        plot_comparison(
            test_tensor, bnb_result, custom_result, f"Comparison for {shape} tensor"
        )

# test_different_shapes()
"""
