import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tabulate import tabulate

# Set a fresh and consistent plot style
sns.set_theme(style="whitegrid")
# Use a vibrant color palette
PALETTE = (
    "bright"  # Alternative options: "colorblind", "deep", "viridis", "plasma", "tab10"
)


def display_precision_info():
    """Display basic information about different floating-point precisions using tabulate."""
    # Data for precision types
    data = [
        ["float64", 64, 11, 52, "~15-17", "±2^-1022 to ±2^1023"],
        ["float32", 32, 8, 23, "~7-8", "±2^-126 to ±2^127"],
        ["float16", 16, 5, 10, "~3-4", "±2^-14 to ±2^15"],
        ["float8", 8, 4, 3, "~1", "±2^-6 to ±2^7"],
    ]

    # Column headers
    headers = [
        "Type",
        "Total Bits",
        "Exponent Bits",
        "Mantissa Bits",
        "Digits",
        "Approx. Range",
    ]

    # Convert to DataFrame for better display
    df = pd.DataFrame(data, columns=headers)

    # Display as a nicely formatted table with tabulate
    print(tabulate(df, headers="keys", tablefmt="pretty", showindex=False))

    # Return data as a DataFrame for potential further analysis
    return df


def compute_function_precision(
    torch_func, func_name=None, x_range=(-5, 5), num_points=1000
):
    """
    Compute a mathematical function across different precision types.

    Args:
        torch_func: PyTorch function (e.g., torch.tanh, torch.sin)
        func_name: Name of the function for plot labels (defaults to function's __name__)
        x_range: Tuple of (min_x, max_x) for input values
        num_points: Number of points to evaluate

    Returns:
        Dictionary with input values, reference values, results, and errors
    """
    # Use the function's name if not provided
    if func_name is None:
        func_name = torch_func.__name__

    # Adjust range for log function to avoid negative/zero inputs
    if torch_func == torch.log:
        x_values = np.linspace(max(0.01, x_range[0]), x_range[1], num_points)
    else:
        x_values = np.linspace(x_range[0], x_range[1], num_points)

    # Create reference values using torch.float64 for high precision
    x_reference = torch.tensor(x_values, dtype=torch.float64)
    reference = torch_func(x_reference).numpy()

    # Create tensors with different precisions
    results = {}

    # Float64 implementation (same as reference)
    results["float64"] = reference.copy()

    # Float32 implementation
    x_float32 = torch.tensor(x_values, dtype=torch.float32)
    results["float32"] = torch_func(x_float32).numpy()

    # Float16 implementation
    x_float16 = torch.tensor(x_values, dtype=torch.float16)
    results["float16"] = torch_func(x_float16).numpy()

    # Float8 implementation (using float8_e4m3fn type for PyTorch)
    x_float8 = torch.tensor(x_values, dtype=torch.float8_e4m3fn)
    # Convert to float32 for computation
    x_float8_as_float32 = x_float8.to(torch.float32)
    # Compute function
    result_float32 = torch_func(x_float8_as_float32)
    # Quantize result back to float8 to simulate full float8 operation
    result_float8 = torch.tensor(result_float32.numpy(), dtype=torch.float8_e4m3fn)
    results["float8"] = result_float8.to(torch.float32).numpy()

    # Calculate absolute errors
    errors = {}
    for precision, values in results.items():
        errors[precision] = np.abs(values - reference)

    return {
        "func_name": func_name,
        "x_values": x_values,
        "reference": reference,
        "results": results,
        "errors": errors,
    }


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


def compare_memory_usage():
    """Compare memory usage of different precision types."""
    # Create large tensors of different precisions
    size = 1_000_000  # 1 million elements

    # Create tensors
    tensor_float64 = torch.ones(size, dtype=torch.float64)
    tensor_float32 = torch.ones(size, dtype=torch.float32)
    tensor_float16 = torch.ones(size, dtype=torch.float16)
    tensor_float8 = torch.ones(size, dtype=torch.float8_e4m3fn)

    # Calculate memory usage (in bytes)
    memory_float64 = tensor_float64.element_size() * tensor_float64.nelement()
    memory_float32 = tensor_float32.element_size() * tensor_float32.nelement()
    memory_float16 = tensor_float16.element_size() * tensor_float16.nelement()
    memory_float8 = tensor_float8.element_size() * tensor_float8.nelement()

    # Print results
    print(f"Memory usage for 1 million elements:")
    print(f"float64: {memory_float64/1024/1024:.2f} MB (100%)")
    print(
        f"float32: {memory_float32/1024/1024:.2f} MB ({memory_float32/memory_float64*100:.1f}%)"
    )
    print(
        f"float16: {memory_float16/1024/1024:.2f} MB ({memory_float16/memory_float64*100:.1f}%)"
    )
    print(
        f"float8: {memory_float8/1024/1024:.2f} MB ({memory_float8/memory_float64*100:.1f}%)"
    )

    # Create visualization
    precisions = ["float64", "float32", "float16", "float8"]
    memory_usage = [
        memory_float64 / (1024 * 1024),  # Convert to MB
        memory_float32 / (1024 * 1024),
        memory_float16 / (1024 * 1024),
        memory_float8 / (1024 * 1024),
    ]

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


def analyze_floating_point_precision(functions=None, include_memory_comparison=True):
    """
    Master function that orchestrates the analysis of floating-point precision.

    Args:
        functions: List of PyTorch functions to analyze. If None, defaults to [torch.tanh, torch.exp, torch.log]
        include_memory_comparison: Whether to include memory usage comparison (default: True)

    Returns:
        Dictionary containing results from all computations
    """
    # Set default plot settings
    set_plot_defaults()

    # Display precision information table
    print("Displaying precision characteristics:")
    precision_df = display_precision_info()
    print("\n")

    # Default functions if none provided
    if functions is None:
        functions = [
            (torch.tanh, None, (-5, 5)),
            (torch.exp, None, (-5, 5)),
            (torch.log, None, (0.1, 5)),
        ]

    # Process each function
    results = {}
    for func_info in functions:
        # Handle different ways to specify functions
        if isinstance(func_info, tuple):
            if len(func_info) == 3:
                func, name, x_range = func_info
            elif len(func_info) == 2:
                func, name = func_info
                x_range = (-5, 5)
            else:
                func = func_info[0]
                name = None
                x_range = (-5, 5)
        else:
            func = func_info
            name = None
            x_range = (-5, 5)

        # Get function name for display
        display_name = name if name else func.__name__
        print(f"\nAnalyzing {display_name} function...")

        # Compute precision for this function
        data = compute_function_precision(func, func_name=name, x_range=x_range)

        # Plot the results
        plot_function_subplots(data)

        # Store results
        results[display_name] = data

    # Compare memory usage
    if include_memory_comparison:
        print("\nComparing memory usage across precision types:")
        compare_memory_usage()

    return {"precision_info": precision_df, "function_results": results}
