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
from src.plot.plot_dtype import (
    analyze_floating_point_precision,
    set_plot_defaults,
)

# Set plot defaults
set_plot_defaults()

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
