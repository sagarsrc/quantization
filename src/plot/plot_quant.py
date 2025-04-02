import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_comparison(
    original,
    bnb_dequant,
    custom_dequant,
    title="Comparison of Quantization Methods",
    sample_size=100,
):
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
    sns.lineplot(
        x=sample_indices,
        y=original_flat[:sample_size],
        ax=axes[0, 0],
        label="Original",
        linewidth=2,
    )
    sns.lineplot(
        x=sample_indices,
        y=bnb_flat[:sample_size],
        ax=axes[0, 0],
        label="BnB",
        alpha=0.8,
        linewidth=1.5,
    )
    axes[0, 0].set_title("Original vs BnB")
    axes[0, 0].legend(loc="best")

    # Second plot with seaborn
    sns.lineplot(
        x=sample_indices,
        y=original_flat[:sample_size],
        ax=axes[0, 1],
        label="Original",
        linewidth=2,
    )
    sns.lineplot(
        x=sample_indices,
        y=custom_flat[:sample_size],
        ax=axes[0, 1],
        label="Custom",
        alpha=0.8,
        linewidth=1.5,
    )
    axes[0, 1].set_title("Original vs Custom")
    axes[0, 1].legend(loc="best")

    # Plot error histograms with KDE
    bnb_errors = original_flat - bnb_flat
    custom_errors = original_flat - custom_flat

    sns.histplot(
        bnb_errors, ax=axes[1, 0], kde=True, stat="density", color="blue", alpha=0.7
    )
    axes[1, 0].set_title(
        f"BnB Error Distribution\nMAE: {bnb_mae:.6f}, MSE: {bnb_mse:.6f}"
    )

    sns.histplot(
        custom_errors, ax=axes[1, 1], kde=True, stat="density", color="blue", alpha=0.7
    )
    axes[1, 1].set_title(
        f"Custom Error Distribution\nMAE: {custom_mae:.6f}, MSE: {custom_mse:.6f}"
    )

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_heatmap_comparison(
    original,
    bnb_dequant,
    custom_dequant,
    title="Heatmap Comparison",
    sample_shape=(16, 16),
):
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
        orig_sample = original[: sample_shape[0], : sample_shape[1]].cpu().numpy()
        bnb_sample = bnb_dequant[: sample_shape[0], : sample_shape[1]].cpu().numpy()
        custom_sample = (
            custom_dequant[: sample_shape[0], : sample_shape[1]].cpu().numpy()
        )
    else:
        sample_size = min(original.numel(), sample_shape[0] * sample_shape[1])
        orig_sample = original[:sample_size].reshape(sample_shape).cpu().numpy()
        bnb_sample = bnb_dequant[:sample_size].reshape(sample_shape).cpu().numpy()
        custom_sample = custom_dequant[:sample_size].reshape(sample_shape).cpu().numpy()

    # Calculate error maps
    bnb_error = np.abs(orig_sample - bnb_sample)
    custom_error = np.abs(orig_sample - custom_sample)
    error_diff = (
        custom_error - bnb_error
    )  # Difference in errors (positive = custom is worse)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Plot the original values and dequantized values
    sns.heatmap(orig_sample, ax=axes[0, 0], cmap="viridis", cbar_kws={"label": "Value"})
    axes[0, 0].set_title("Original Values")

    sns.heatmap(bnb_sample, ax=axes[0, 1], cmap="viridis", cbar_kws={"label": "Value"})
    axes[0, 1].set_title("BnB Dequantized")

    sns.heatmap(
        custom_sample, ax=axes[0, 2], cmap="viridis", cbar_kws={"label": "Value"}
    )
    axes[0, 2].set_title("Custom Dequantized")

    # Plot the error heatmaps
    max_error = max(np.max(bnb_error), np.max(custom_error))

    sns.heatmap(
        bnb_error,
        ax=axes[1, 0],
        cmap="Reds",
        vmin=0,
        vmax=max_error,
        cbar_kws={"label": "Absolute Error"},
    )
    axes[1, 0].set_title(f"BnB Error (MAE: {np.mean(bnb_error):.4f})")

    sns.heatmap(
        custom_error,
        ax=axes[1, 1],
        cmap="Reds",
        vmin=0,
        vmax=max_error,
        cbar_kws={"label": "Absolute Error"},
    )
    axes[1, 1].set_title(f"Custom Error (MAE: {np.mean(custom_error):.4f})")

    # Use diverging colormap for error difference
    max_diff = max(abs(np.min(error_diff)), abs(np.max(error_diff)))
    sns.heatmap(
        error_diff,
        ax=axes[1, 2],
        cmap="coolwarm",
        vmin=-max_diff,
        vmax=max_diff,
        cbar_kws={"label": "Error Difference"},
    )
    axes[1, 2].set_title("Error Difference (Custom - BnB)")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

    return orig_sample, bnb_sample, custom_sample, bnb_error, custom_error


def plot_quant_comparison(
    matrix,
    bnb_dequantized,
    custom_dequantized,
    bnb_errors,
    custom_errors,
    bnb_mae,
    bnb_mse,
    custom_mae,
    custom_mse,
    sample_size=50,
):
    """
    Plot comparison of original matrix vs BnB and custom dequantized values,
    along with error distributions.

    Args:
        matrix: Original tensor
        bnb_dequantized: Tensor dequantized with BnB
        custom_dequantized: Tensor dequantized with custom implementation
        bnb_errors: Error tensor for BnB implementation
        custom_errors: Error tensor for custom implementation
        bnb_mae: Mean Absolute Error for BnB implementation
        bnb_mse: Mean Squared Error for BnB implementation
        custom_mae: Mean Absolute Error for custom implementation
        custom_mse: Mean Squared Error for custom implementation
        sample_size: Number of values to plot in the line plots

    Returns:
        None
    """
    # Sample data for visualization
    matrix_sample = matrix.flatten()[:sample_size].cpu().numpy()
    bnb_sample = bnb_dequantized.flatten()[:sample_size].cpu().numpy()
    custom_sample = custom_dequantized.flatten()[:sample_size].cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Value comparison plots using seaborn
    sample_indices = range(sample_size)
    sns.lineplot(
        x=sample_indices, y=matrix_sample, ax=axes[0, 0], label="Original", linewidth=2
    )
    sns.lineplot(
        x=sample_indices,
        y=bnb_sample,
        ax=axes[0, 0],
        label="BnB",
        alpha=0.8,
        linewidth=1.5,
    )
    axes[0, 0].set_title("Original vs BnB")
    axes[0, 0].legend()

    sns.lineplot(
        x=sample_indices, y=matrix_sample, ax=axes[0, 1], label="Original", linewidth=2
    )
    sns.lineplot(
        x=sample_indices,
        y=custom_sample,
        ax=axes[0, 1],
        label="Custom",
        alpha=0.8,
        linewidth=1.5,
    )
    axes[0, 1].set_title("Original vs Custom")
    axes[0, 1].legend()

    # Error distribution plots with seaborn
    bnb_errors_flat = bnb_errors.flatten().cpu().numpy()
    custom_errors_flat = custom_errors.flatten().cpu().numpy()

    sns.histplot(
        bnb_errors_flat,
        ax=axes[1, 0],
        kde=True,
        stat="density",
        color="blue",
        alpha=0.7,
    )
    axes[1, 0].set_title(f"BnB Error (MAE: {bnb_mae:.6f}, MSE: {bnb_mse:.6f})")

    sns.histplot(
        custom_errors_flat,
        ax=axes[1, 1],
        kde=True,
        stat="density",
        color="blue",
        alpha=0.7,
    )
    axes[1, 1].set_title(f"Custom Error (MAE: {custom_mae:.6f}, MSE: {custom_mse:.6f})")

    plt.tight_layout()
    plt.show()


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
    sns.histplot(bnb_absmax, kde=True, color="blue", alpha=0.5, label="Original absmax")
    sns.histplot(
        custom_absmax, kde=True, color="red", alpha=0.5, label="Scaled absmax (÷12)"
    )
    plt.title("Distribution of absmax values across blocks", fontsize=16)
    plt.xlabel("absmax value")
    plt.ylabel("Frequency")
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
    plt.plot(x_ref, y_ref, "r--", label="y = x/12")

    plt.title("Scaling Relationship between absmax values", fontsize=16)
    plt.xlabel("Original absmax")
    plt.ylabel("Scaled absmax (÷12)")
    plt.legend()
    plt.grid(True)
    plt.show()

    return bnb_absmax, custom_absmax
