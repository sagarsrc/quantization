#%%
import notebook_setup
import torch
import bitsandbytes.functional as F
from src.utils.bnb_functional import quantize_fp4, dequantize_fp4
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# Set seaborn style for better visualizations
sns.set_style("whitegrid")
sns.set_context("talk")

#%% [markdown]
# # FP4 Quantization and Dequantization Analysis
#
# This notebook compares the BitsAndBytes (BnB) and custom implementations of FP4 quantization.
# We'll analyze:
# 1. Quantization accuracy
# 2. Error distributions
# 3. Visual comparisons of the representations

#%%
# Main configuration
device = torch.device("cuda")
blocksize = 64

#%% [markdown]
# ## Basic Testing
#
# First, we'll run a simple test to compare both implementations on a small tensor.

#%%
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

    return A, (A_fp4_bnb, bnb_state, A_dequant_bnb), (A_fp4_custom, custom_state, A_dequant_custom)

# Run basic test
original, bnb_results, custom_results = run_basic_test()

#%% [markdown]
# ## Visual Comparison Methods
#
# The following functions provide visual comparisons between original tensors and their quantized/dequantized versions.
#
# ### Line Plot Comparison
#
# This plot shows how well the dequantized values match the original tensor values. The closer the lines, the better.
# - Top row: Direct value comparison between original and dequantized values
# - Bottom row: Error distribution, showing how errors are distributed (ideally centered around zero)

#%%
def plot_comparison(original, bnb_dequant, custom_dequant, title="Comparison of Quantization Methods",
                   sample_size=100):
    """
    Create visual comparison between original tensor and both dequantized versions

    Args:
        original: Original tensor
        bnb_dequant: Tensor dequantized with BnB
        custom_dequant: Tensor dequantized with custom implementation
        title: Plot title
        sample_size: Number of values to plot

    Returns:
        None
    """
    # Flatten tensors for visualization
    original_flat = original.flatten().cpu().numpy()
    bnb_flat = bnb_dequant.flatten().cpu().numpy()
    custom_flat = custom_dequant.flatten().cpu().numpy()

    # Calculate error metrics
    bnb_mae = torch.mean(torch.abs(original - bnb_dequant)).item()
    custom_mae = torch.mean(torch.abs(original - custom_dequant)).item()
    bnb_mse = torch.mean((original - bnb_dequant) ** 2).item()
    custom_mse = torch.mean((original - custom_dequant) ** 2).item()

    # Create plots with seaborn styling
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot original vs. dequantized values
    sample_indices = range(min(sample_size, len(original_flat)))

    # First plot with seaborn
    sns.lineplot(x=sample_indices, y=original_flat[:sample_size], ax=axes[0, 0],
                label='Original', linewidth=2)
    sns.lineplot(x=sample_indices, y=bnb_flat[:sample_size], ax=axes[0, 0],
                label='BnB', alpha=0.8, linewidth=1.5)
    axes[0, 0].set_title('Original vs BnB')
    axes[0, 0].legend(loc='best')

    # Second plot with seaborn
    sns.lineplot(x=sample_indices, y=original_flat[:sample_size], ax=axes[0, 1],
                label='Original', linewidth=2)
    sns.lineplot(x=sample_indices, y=custom_flat[:sample_size], ax=axes[0, 1],
                label='Custom', alpha=0.8, linewidth=1.5)
    axes[0, 1].set_title('Original vs Custom')
    axes[0, 1].legend(loc='best')

    # Plot error histograms with KDE
    bnb_errors = (original_flat - bnb_flat)
    custom_errors = (original_flat - custom_flat)

    sns.histplot(bnb_errors, ax=axes[1, 0], kde=True, stat="density", color="blue", alpha=0.7)
    axes[1, 0].set_title(f'BnB Error Distribution\nMAE: {bnb_mae:.6f}, MSE: {bnb_mse:.6f}')

    sns.histplot(custom_errors, ax=axes[1, 1], kde=True, stat="density", color="blue", alpha=0.7)
    axes[1, 1].set_title(f'Custom Error Distribution\nMAE: {custom_mae:.6f}, MSE: {custom_mse:.6f}')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

#%% [markdown]
# ### Heatmap Comparison
#
# Heatmaps provide spatial insights into quantization errors:
# - Top row: Values visualization (original, BnB dequantized, custom dequantized)
# - Bottom left: BnB error magnitude (brighter = higher error)
# - Bottom middle: Custom error magnitude (brighter = higher error)
# - Bottom right: Error difference (blue = BnB better, red = custom better)

#%%
def plot_heatmap_comparison(original, bnb_dequant, custom_dequant, title="Heatmap Comparison",
                           sample_shape=(16, 16)):
    """
    Create heatmaps to compare the original values with BnB and custom implementation results.
    Also visualize the error heatmaps.

    Args:
        original: Original tensor
        bnb_dequant: Tensor dequantized with BnB
        custom_dequant: Tensor dequantized with custom implementation
        title: Title for the plot
        sample_shape: Shape of the sample to visualize (default: 16x16 block)

    Returns:
        Tuple of numpy arrays containing samples and error matrices
    """
    # Take a sample portion for better visualization
    if len(original.shape) > 1:
        orig_sample = original[:sample_shape[0], :sample_shape[1]].cpu().numpy()
        bnb_sample = bnb_dequant[:sample_shape[0], :sample_shape[1]].cpu().numpy()
        custom_sample = custom_dequant[:sample_shape[0], :sample_shape[1]].cpu().numpy()
    else:
        sample_size = min(original.numel(), sample_shape[0] * sample_shape[1])
        orig_sample = original[:sample_size].reshape(sample_shape).cpu().numpy()
        bnb_sample = bnb_dequant[:sample_size].reshape(sample_shape).cpu().numpy()
        custom_sample = custom_dequant[:sample_size].reshape(sample_shape).cpu().numpy()

    # Calculate error maps
    bnb_error = np.abs(orig_sample - bnb_sample)
    custom_error = np.abs(orig_sample - custom_sample)
    error_diff = custom_error - bnb_error  # Difference in errors (positive = custom is worse)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Plot the original values and dequantized values
    sns.heatmap(orig_sample, ax=axes[0, 0], cmap='viridis', cbar_kws={'label': 'Value'})
    axes[0, 0].set_title('Original Values')

    sns.heatmap(bnb_sample, ax=axes[0, 1], cmap='viridis', cbar_kws={'label': 'Value'})
    axes[0, 1].set_title('BnB Dequantized')

    sns.heatmap(custom_sample, ax=axes[0, 2], cmap='viridis', cbar_kws={'label': 'Value'})
    axes[0, 2].set_title('Custom Dequantized')

    # Plot the error heatmaps
    max_error = max(np.max(bnb_error), np.max(custom_error))

    sns.heatmap(bnb_error, ax=axes[1, 0], cmap='Reds', vmin=0, vmax=max_error,
                cbar_kws={'label': 'Absolute Error'})
    axes[1, 0].set_title(f'BnB Error (MAE: {np.mean(bnb_error):.4f})')

    sns.heatmap(custom_error, ax=axes[1, 1], cmap='Reds', vmin=0, vmax=max_error,
                cbar_kws={'label': 'Absolute Error'})
    axes[1, 1].set_title(f'Custom Error (MAE: {np.mean(custom_error):.4f})')

    # Use diverging colormap for error difference
    max_diff = max(abs(np.min(error_diff)), abs(np.max(error_diff)))
    sns.heatmap(error_diff, ax=axes[1, 2], cmap='coolwarm', vmin=-max_diff, vmax=max_diff,
                cbar_kws={'label': 'Error Difference'})
    axes[1, 2].set_title('Error Difference (Custom - BnB)')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

    return orig_sample, bnb_sample, custom_sample, bnb_error, custom_error

#%% [markdown]
# ## Comprehensive Implementation Comparison
#
# This function performs detailed analysis between the two implementations on a given tensor,
# including:
# - Timing measurements
# - Error metrics (MAE, MSE)
# - Memory efficiency
# - Implementation similarity

#%%
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
        padded_tensor = torch.zeros(padded_shape, dtype=test_tensor.dtype, device=test_tensor.device)
        padded_tensor[..., :test_tensor.shape[-1]] = test_tensor
        test_tensor = padded_tensor
        print(f"Tensor padded to shape {test_tensor.shape} to ensure divisibility by blocksize {blocksize}")

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
        'Metric': ['Time (ms)', 'MAE', 'MSE', 'Size (bytes)', 'Compression Ratio'],
        'BnB': [bnb_time, bnb_mae, bnb_mse, bnb_bytes, original_bytes/bnb_bytes],
        'Custom': [custom_time, custom_mae, custom_mse, custom_bytes, original_bytes/custom_bytes],
        'Ratio/Diff': [custom_time/bnb_time, custom_mae/bnb_mae, custom_mse/bnb_mse,
                        custom_bytes/bnb_bytes, (original_bytes/custom_bytes)/(original_bytes/bnb_bytes)]
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

#%% [markdown]
# ## Matrix Quantization Analysis
#
# This function tests both implementations on a random matrix and analyzes quantization/dequantization performance.
#
# The plots show:
# 1. **Value Plots**: Original vs quantized values sample - how closely the dequantized values follow the original
# 2. **Error Distributions**: How the errors are distributed - a narrower distribution around zero indicates better performance

#%%
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
    implementation_diff = torch.mean(torch.abs(bnb_dequantized - custom_dequantized)).item()

    # Print comparison results
    print(f"Matrix shape: {matrix.shape}")
    print(f"BnB MAE: {bnb_mae:.6f}, MSE: {bnb_mse:.6f}")
    print(f"Custom MAE: {custom_mae:.6f}, MSE: {custom_mse:.6f}")
    print(f"Implementation difference: {implementation_diff:.6f}")

    # Plot comparison
    sample_size = 50
    matrix_sample = matrix.flatten()[:sample_size].cpu().numpy()
    bnb_sample = bnb_dequantized.flatten()[:sample_size].cpu().numpy()
    custom_sample = custom_dequantized.flatten()[:sample_size].cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Value comparison plots using seaborn
    sample_indices = range(sample_size)
    sns.lineplot(x=sample_indices, y=matrix_sample, ax=axes[0, 0], label='Original', linewidth=2)
    sns.lineplot(x=sample_indices, y=bnb_sample, ax=axes[0, 0], label='BnB', alpha=0.8, linewidth=1.5)
    axes[0, 0].set_title('Original vs BnB')
    axes[0, 0].legend()

    sns.lineplot(x=sample_indices, y=matrix_sample, ax=axes[0, 1], label='Original', linewidth=2)
    sns.lineplot(x=sample_indices, y=custom_sample, ax=axes[0, 1], label='Custom', alpha=0.8, linewidth=1.5)
    axes[0, 1].set_title('Original vs Custom')
    axes[0, 1].legend()

    # Error distribution plots with seaborn
    bnb_errors = (matrix - bnb_dequantized).flatten().cpu().numpy()
    custom_errors = (matrix - custom_dequantized).flatten().cpu().numpy()

    sns.histplot(bnb_errors, ax=axes[1, 0], kde=True, stat="density", color="blue", alpha=0.7)
    axes[1, 0].set_title(f'BnB Error (MAE: {bnb_mae:.6f})')

    sns.histplot(custom_errors, ax=axes[1, 1], kde=True, stat="density", color="blue", alpha=0.7)
    axes[1, 1].set_title(f'Custom Error (MAE: {custom_mae:.6f})')

    plt.tight_layout()
    plt.show()

    return matrix, (bnb_quantized, bnb_dequantized), (custom_quantized, custom_dequantized)

#%% [markdown]
# ## AbsMax Scaling Analysis
#
# This function visualizes how absmax values are calculated in both implementations.
#
# The plots show:
# 1. **Distribution**: How absmax values are distributed across blocks
# 2. **Scaling Relationship**: The relationship between BnB and custom absmax values (should follow y = x/12 line)

#%%
def plot_absmax_comparison(test_tensor, blocksize=64):
    """
    Visualize how absmax values are calculated in both implementations

    Args:
        test_tensor: Tensor to analyze
        blocksize: Block size for quantization

    Returns:
        Tuple of numpy arrays with absmax values from both approaches
    """
    # Reshape tensor into blocks
    blocks = test_tensor.reshape(-1, blocksize)

    # Calculate absmax according to both methods
    bnb_absmax = blocks.abs().max(dim=1).values.float().cpu().numpy()
    custom_absmax = (blocks.abs().max(dim=1).values / 12.0).float().cpu().numpy()

    # Plot the distributions
    plt.figure(figsize=(12, 6))
    sns.histplot(bnb_absmax, kde=True, color='blue', alpha=0.5, label='Original absmax')
    sns.histplot(custom_absmax, kde=True, color='red', alpha=0.5, label='Scaled absmax (÷12)')
    plt.title('Distribution of absmax values across blocks', fontsize=16)
    plt.xlabel('absmax value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Create scatter plot to compare scaling
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=bnb_absmax, y=custom_absmax, alpha=0.7)

    # Add reference line (y = x/12)
    max_val = max(bnb_absmax.max(), custom_absmax.max() * 12)
    x_ref = np.linspace(0, max_val, 100)
    y_ref = x_ref / 12.0
    plt.plot(x_ref, y_ref, 'r--', label='y = x/12')

    plt.title('Scaling Relationship between absmax values', fontsize=16)
    plt.xlabel('Original absmax')
    plt.ylabel('Scaled absmax (÷12)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return bnb_absmax, custom_absmax

#%% [markdown]
# ## Complete Comparison Suite
#
# This function runs a comprehensive comparison suite on multiple tensor shapes.
#
# The analysis includes:
# - Performance metrics on reference tensor
# - Visual comparisons of quantization quality
# - Evaluation across different tensor shapes to measure consistency

#%%
def run_comparison_suite(shapes_to_test=None):
    """
    Run a comprehensive comparison suite on multiple tensor shapes

    Args:
        shapes_to_test: List of tensor shapes to test

    Returns:
        None
    """
    if shapes_to_test is None:
        shapes_to_test = [
            (128, 64),         # Standard rectangular matrix
            # (256, 256),      # Medium square matrix
            # (1024, 64),      # Tall matrix
            # (64, 1024)       # Wide matrix
        ]

    # Create a mid-sized test tensor for detailed analysis
    torch.manual_seed(42)
    test_tensor = torch.randn(128, 64, dtype=torch.float16).to(device)

    print("\n===== DETAILED ANALYSIS ON REFERENCE TENSOR =====")
    # Run comprehensive comparison on main test tensor
    bnb_result, custom_result = compare_implementations(test_tensor)

    # Show visual comparison
    plot_comparison(test_tensor, bnb_result, custom_result, "Comparison of Quantization Methods")

    # Generate heatmap visualization
    orig_sample, bnb_sample, custom_sample, bnb_error, custom_error = plot_heatmap_comparison(
        test_tensor, bnb_result, custom_result, "Heatmap Comparison"
    )

    # Compare FP4 quantization values
    matrix, (bnb_quantized, bnb_dequantized), (custom_quantized, custom_dequantized) = compare_quant_values()

    # Analyze absmax scaling
    original_absmax, scaled_absmax = plot_absmax_comparison(test_tensor)

    # Test with different shapes
    print("\n===== TESTING DIFFERENT TENSOR SHAPES =====")
    for shape in shapes_to_test:
        if shape == (128, 64):  # Skip if already tested
            continue

        print(f"\n{'='*50}")
        print(f"Testing with shape: {shape}")
        print(f"{'='*50}")
        # All these shapes have total elements divisible by 64
        test_tensor = torch.randn(*shape, dtype=torch.float16).to(device)
        bnb_result, custom_result = compare_implementations(test_tensor)
        # Plot comparison for this shape
        plot_comparison(test_tensor, bnb_result, custom_result, f"Comparison for {shape} tensor")

#%% [markdown]
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

#%%
# Run the full comparison suite
run_comparison_suite()
