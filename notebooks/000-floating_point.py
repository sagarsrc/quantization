# %% [markdown]
# # Float Precision Comparison
#
# This notebook demonstrates the differences between float64, float32, float16, and float8
# data types using various mathematical functions.

# %%
import notebook_setup
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from src.plot.plot_dtype import (
    set_plot_defaults,
    plot_function_subplots,
    plot_error_comparison,
    plot_error_metrics,
)

# Set plot defaults
set_plot_defaults()

# %%
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

# %%
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

# %%
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

    # Create visualization for memory usage comparison
    from src.plot.plot_dtype import plot_memory_usage
    precisions = ["float64", "float32", "float16", "float8"]
    memory_usage = [
        memory_float64 / (1024 * 1024),  # Convert to MB
        memory_float32 / (1024 * 1024),
        memory_float16 / (1024 * 1024),
        memory_float8 / (1024 * 1024),
    ]
    plot_memory_usage(precisions, memory_usage)

# %%
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
        data = compute_function_precision(func, func_name=display_name, x_range=x_range)

        # Plot the results
        plot_function_subplots(data)

        # Store results
        results[display_name] = data

    # Compare memory usage
    if include_memory_comparison:
        print("\nComparing memory usage across precision types:")
        compare_memory_usage()

    return {"precision_info": precision_df, "function_results": results}

# %% [markdown]
# ## Float Precision Analysis
#
# This analysis includes:
# 1. Basic characteristics of different floating-point formats
# 2. Comparison of mathematical functions across precision types
# 3. Visualization of results with different functions (tanh, exp, log)
# 4. Memory usage comparison across precision types

# %%
# Run the complete floating-point precision analysis
functions_to_analyze = [
    (torch.tanh, None, (-5, 5)),
    (torch.exp, None, (-5, 5)),
    # (torch.log, None, (0.1, 5)),
]

results = analyze_floating_point_precision(functions=functions_to_analyze)

# %%

# %% [markdown]
# ## Conclusion
#
# This notebook has demonstrated the precision differences between float64, float32, float16, and float8
# representations using various mathematical functions:
#
# 1. **Accuracy**: Lower precision types show progressively larger errors in representing function values.
#
# 2. **Error Distribution**: Errors are not uniform across the input range; they tend to be higher
#    in regions where the function changes rapidly.
#
# 3. **Memory Usage**: Lower precision types use proportionally less memory, with float8 using
#    just 12.5% of the memory required by float64.
#
# In the context of LLM quantization, this demonstrates why using lower-precision formats can
# dramatically reduce model size and memory requirements, but at the cost of increased numerical error.
# This error must be carefully managed through techniques like quantization-aware training and
# keeping certain operations (like accumulation) in higher precision.
