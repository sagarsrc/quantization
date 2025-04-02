import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set a fresh and consistent plot style
sns.set_theme(style="whitegrid")
# Use a vibrant color palette
PALETTE = (
    "bright"  # Alternative options: "colorblind", "deep", "viridis", "plasma", "tab10"
)


def plot_function_subplots(data):
    """
    Create subplots to compare function implementations with different precisions.
    Each precision gets its own subplot compared against the reference.

    Args:
        data: Dictionary with input values, reference results, and errors
    """
    x_values = data["x_values"]
    reference = data["reference"]
    results = data["results"]
    func_name = data["func_name"]

    # Create a 2x2 grid of subplots for the different precision types
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    axes = axes.flatten()

    # Colors and precisions
    precisions = ["float64", "float32", "float16", "float8"]
    # Use the globally defined color palette
    colors = sns.color_palette(PALETTE, 4)

    # Plot each precision type against the reference
    for i, (precision, color) in enumerate(zip(precisions, colors)):
        ax = axes[i]
        # Plot reference with some transparency
        ax.plot(
            x_values,
            reference,
            color="black",
            linestyle="-",
            label="Reference",
            alpha=0.4,
            linewidth=2,
        )
        # Plot precision result with dashed line
        ax.plot(
            x_values,
            results[precision],
            color=color,
            linestyle="--",
            label=precision,
            linewidth=2,
        )

        ax.set_title(f"{func_name} Function: {precision} vs Reference")
        ax.set_xlabel("x")
        ax.set_ylabel(f"{func_name}(x)")
        ax.legend()

    plt.tight_layout(pad=2.0)
    plt.show()
    plt.close(fig)

    # Now plot errors in one combined plot
    plot_error_comparison(data)


def plot_error_comparison(data):
    """Plot error comparison for all precisions in a single plot."""
    x_values = data["x_values"]
    errors = data["errors"]
    func_name = data["func_name"]

    # Create figure for error comparison
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    # Use the globally defined color palette
    colors = sns.color_palette(PALETTE, 4)

    # Plot errors for each precision type
    precisions = ["float64", "float32", "float16", "float8"]
    for i, precision in enumerate(precisions):
        ax.semilogy(
            x_values,
            errors[precision],
            color=colors[i],
            linestyle="-",
            label=f"{precision} error",
            linewidth=1.5,
        )

    ax.set_title(f"Absolute Errors in {func_name} Calculation (Log Scale)")
    ax.set_xlabel("x")
    ax.set_ylabel("Absolute Error")
    ax.legend()

    plt.tight_layout(pad=1.0)
    plt.show()
    plt.close(fig)

    # Create bar chart of maximum and average errors
    plot_error_metrics(data)


def plot_error_metrics(data):
    """Create bar chart showing error metrics."""
    errors = data["errors"]
    func_name = data["func_name"]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    precisions = ["float64", "float32", "float16", "float8"]
    max_errors = [np.max(errors[p]) for p in precisions]
    avg_errors = [np.mean(errors[p]) for p in precisions]

    # Create a DataFrame for better seaborn integration
    df_max = pd.DataFrame(
        {
            "Precision": precisions,
            "Error": max_errors,
            "Type": ["Maximum"] * len(precisions),
        }
    )
    df_avg = pd.DataFrame(
        {
            "Precision": precisions,
            "Error": avg_errors,
            "Type": ["Average"] * len(precisions),
        }
    )
    df = pd.concat([df_max, df_avg])

    # Use seaborn for bar plot with proper hue parameter
    ax = sns.barplot(data=df, x="Precision", y="Error", hue="Type", palette=PALETTE)

    plt.yscale("log")
    plt.title(f"Error Metrics for {func_name} Implementation by Precision Type")
    plt.xlabel("Precision Type")
    plt.ylabel("Error (Log Scale)")

    # Add value labels - only if the value is significant
    for i, v in enumerate(max_errors):
        if v > 1e-10:  # Only show label if value is significant
            plt.text(i - 0.2, v * 1.1, f"{v:.2e}", ha="center", fontsize=9)

    for i, v in enumerate(avg_errors):
        if v > 1e-10:  # Only show label if value is significant
            plt.text(i + 0.2, v * 1.1, f"{v:.2e}", ha="center", fontsize=9)

    # Set a minimum y limit to avoid showing extremely small values
    plt.ylim(bottom=1e-10)

    plt.tight_layout(pad=1.5)
    plt.show()
    plt.close(fig)


def plot_memory_usage(precisions, memory_usage):
    """
    Plot memory usage comparison between different precision types.

    Args:
        precisions: List of precision names (e.g., ["float64", "float32", "float16", "float8"])
        memory_usage: List of memory usage values in MB
    """
    # Create figure with appropriate height
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

    # Use matplotlib's bar function directly to avoid seaborn warning
    colors = sns.color_palette(PALETTE, 4)
    bars = ax.bar(precisions, memory_usage, color=colors, width=0.6)

    # Set y-axis limits to provide more space for labels
    # Add 20% padding above the highest bar
    max_memory = max(memory_usage)
    ax.set_ylim(0, max_memory * 1.3)  # Increased padding for labels above bars

    # Add percentage labels consistently above all bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        percentage = memory_usage[i] / memory_usage[0] * 100

        # Position all labels above the bars
        text_y = height + (max_memory * 0.03)  # Small offset above each bar

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            text_y,
            f"{height:.2f} MB\n({percentage:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Customize plot appearance
    plt.title("Memory Usage Comparison (1M elements)", fontsize=14, pad=20)
    plt.ylabel("Memory (MB)", fontsize=12)
    plt.xlabel("Precision Type", fontsize=12)

    # Add grid only on y-axis with lighter color
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add a thin horizontal line at y=0
    ax.axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout(pad=1.5)
    plt.show()
    plt.close(fig)


# Set default matplotlib settings
def set_plot_defaults():
    """Set default matplotlib settings for plots."""
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
