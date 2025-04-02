########################################################################
# Note:
# The following code is borrowed from bitsandbytes/functional.py file
# and modified to be used in this project
########################################################################
import torch
import numpy as np
from typing import Optional, Tuple, Any, Dict, Union, Sequence


# FP4 quantization values - updated to match bitsandbytes implementation
# These values are from the bitsandbytes FP4 data type implementation
FP4_VALUES = torch.tensor([
    0.0,         # 0000 (0)
    0.0625,      # 0001 (1) - smallest positive value
    8.0,         # 0010 (2)
    12.0,        # 0011 (3)
    4.0,         # 0100 (4)
    6.0,         # 0101 (5)
    2.0,         # 0110 (6)
    3.0,         # 0111 (7) - largest positive normal
    0.0,         # 1000 (8) - zero with negative sign bit (treated as 0)
    -0.0625,     # 1001 (9) - smallest negative value
    -8.0,        # 1010 (10)
    -12.0,       # 1011 (11)
    -4.0,        # 1100 (12)
    -6.0,        # 1101 (13)
    -2.0,        # 1110 (14)
    -3.0,        # 1111 (15) - largest negative normal
], dtype=torch.float32)

# Cache FP4 values per device for performance
_FP4_CACHE = {}

# Get device-specific FP4 values with caching
def get_fp4_values(device):
    if device not in _FP4_CACHE:
        _FP4_CACHE[device] = FP4_VALUES.to(device)
    return _FP4_CACHE[device]

class QuantState:
    """Container for quantization state components"""

    def __init__(
        self,
        absmax: torch.Tensor,
        shape: Optional[torch.Size] = None,
        code: Optional[torch.Tensor] = None,
        blocksize: Optional[int] = None,
        quant_type: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.absmax = absmax
        self.shape = shape
        self.code = code
        self.dtype = dtype
        self.blocksize = blocksize
        self.quant_type = quant_type

    def to(self, device):
        # Make sure the quantization state is on the right device
        self.code = self.code.to(device)
        self.absmax = self.absmax.to(device)
        return self


def quantize_fp4(
    A: torch.Tensor,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize: int = 64,
) -> Tuple[torch.Tensor, QuantState]:
    """
    Quantize a tensor to 4-bit floating point (FP4).

    Args:
        A: Input tensor to quantize.
        absmax: Optional tensor to store the absmax values. If None, will be computed.
        out: Optional tensor to store the output. If None, a new tensor will be created.
        blocksize: Size of blocks to quantize independently. Default: 64.

    Returns:
        Tuple containing:
            - Quantized tensor with packed 4-bit values
            - QuantState object containing the quantization state for dequantization
    """
    return quantize_4bit(A, absmax, out, blocksize, "fp4")


def quantize_4bit(
    A: torch.Tensor,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize: int = 64,
    quant_type: str = "fp4",
) -> Tuple[torch.Tensor, QuantState]:
    """
    Quantize a tensor to 4-bit with specified quantization type.

    Args:
        A: Input tensor to quantize.
        absmax: Optional tensor to store the absmax values. If None, will be computed.
        out: Optional tensor to store the output. If None, a new tensor will be created.
        blocksize: Size of blocks to quantize independently. Default: 64.
        quant_type: Type of 4-bit quantization to use. Currently only "fp4" is supported.

    Returns:
        Tuple containing:
            - Quantized tensor with packed 4-bit values
            - QuantState object containing the quantization state for dequantization
    """
    assert quant_type == "fp4", "Only fp4 quantization type is supported"

    # Get device and shape information
    device = A.device
    input_shape = A.shape
    n = A.numel()

    # Ensure tensor size is divisible by blocksize for simplicity
    assert n % blocksize == 0, f"Tensor size {n} must be divisible by blocksize {blocksize}"

    # Fast path: reshape to blocks and compute absmax
    # Use torch.view instead of reshape when possible for better performance
    is_contiguous = A.is_contiguous()
    blocks = A.view(-1, blocksize) if is_contiguous else A.reshape(-1, blocksize)
    num_blocks = blocks.shape[0]

    # Calculate absolute maximum value per block with scaling
    if absmax is None:
        with torch.no_grad():
            # Compute block-wise absmax efficiently
            absmax = blocks.abs().max(dim=1).values.div_(12.0).float()
    else:
        # Use the provided absmax tensor
        absmax = absmax.float()

    # Prepare output tensor to hold 4-bit values (packed two per byte)
    if out is None:
        # Allocate half the size (2 values per byte)
        out_size = (n + 1) // 2
        out = torch.zeros(out_size, dtype=torch.uint8, device=device)

    # Get cached FP4 values for this device
    fp4_values = get_fp4_values(device)

    # Process in chunks to reduce memory usage
    # Optimized matrix calculation with preallocated output
    chunk_size = min(32768, n)  # Process up to 32K elements at once
    quantized_indices = torch.empty(n, dtype=torch.uint8, device=device)

    for chunk_start in range(0, n, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n)
        chunk_size_actual = chunk_end - chunk_start

        # Get block indices for this chunk
        block_start = chunk_start // blocksize
        block_end = (chunk_end + blocksize - 1) // blocksize

        # Get corresponding absmax values
        block_indices = torch.arange(block_start, block_end, device=device)
        chunk_absmax = absmax[block_indices].unsqueeze(1).repeat_interleave(
            blocksize, dim=1
        ).flatten()[:chunk_size_actual]

        # Extract chunk data and normalize
        chunk_data = A.flatten()[chunk_start:chunk_end]
        normalized = chunk_data / chunk_absmax

        # Find closest FP4 value
        # Use L1 norm as it's faster than L2 for this application
        dist = torch.abs(normalized.unsqueeze(1) - fp4_values.unsqueeze(0))
        indices = torch.argmin(dist, dim=1).to(torch.uint8)

        # Store results
        quantized_indices[chunk_start:chunk_end] = indices

    # Efficient bit packing using vectorized operations
    # For even indices, shift left by 4
    even_indices = torch.arange(0, n-1, 2, device=device)
    odd_indices = even_indices + 1

    if even_indices.numel() > 0:
        # Get indices
        even_values = quantized_indices[even_indices].to(torch.int32)
        odd_values = quantized_indices[odd_indices].to(torch.int32)

        # Pack values
        packed = torch.bitwise_or(
            torch.bitwise_left_shift(even_values, 4),
            odd_values
        ).to(torch.uint8)

        # Scatter to output tensor
        out.scatter_(0, even_indices // 2, packed)

    # Handle the last element if n is odd
    if n % 2 == 1:
        last_idx = n - 1
        last_value = torch.bitwise_left_shift(
            quantized_indices[last_idx].to(torch.int32), 4
        ).to(torch.uint8)
        out[last_idx // 2] = last_value

    # Create quantization state for dequantization
    state = QuantState(
        absmax=absmax,
        shape=input_shape,
        dtype=A.dtype,
        blocksize=blocksize,
        quant_type=quant_type,
        code=fp4_values
    )

    return out, state


def dequantize_fp4(
    A: torch.Tensor,
    quant_state: Optional[QuantState] = None,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize: int = 64,
) -> torch.Tensor:
    """
    Dequantize a tensor from 4-bit floating point (FP4) representation.

    Args:
        A: Quantized input tensor with packed 4-bit values.
        quant_state: QuantState object containing the quantization state. Required if absmax is None.
        absmax: Tensor containing absmax values. Required if quant_state is None.
        out: Optional tensor to store the output. If None, a new tensor will be created.
        blocksize: Size of blocks that were quantized independently. Default: 64.

    Returns:
        Dequantized tensor with same shape as original input.
    """
    return dequantize_4bit(A, quant_state, absmax, out, blocksize, "fp4")


def dequantize_4bit(
    A: torch.Tensor,
    quant_state: Optional[QuantState] = None,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize: int = 64,
    quant_type: str = "fp4",
) -> torch.Tensor:
    """
    Dequantize a tensor from 4-bit representation.

    Args:
        A: Quantized input tensor with packed 4-bit values.
        quant_state: QuantState object containing the quantization state. Required if absmax is None.
        absmax: Tensor containing absmax values. Required if quant_state is None.
        out: Optional tensor to store the output. If None, a new tensor will be created.
        blocksize: Size of blocks that were quantized independently. Default: 64.
        quant_type: Type of 4-bit quantization used. Currently only "fp4" is supported.

    Returns:
        Dequantized tensor with same shape as original input.
    """
    assert quant_type == "fp4", "Only fp4 quantization type is supported"

    # Get device
    device = A.device

    if quant_state is None:
        assert absmax is not None, "Either quant_state or absmax must be provided"

        # If shape is not provided via quant_state, we need to infer it
        # We know the total number of elements is blocksize * num_blocks
        num_blocks = absmax.numel()
        total_elems = num_blocks * blocksize

        # If out is provided, use its shape, otherwise default to a 1D tensor
        shape = out.shape if out is not None else (total_elems,)

        quant_state = QuantState(
            absmax=absmax,
            shape=shape,
            blocksize=blocksize,
            quant_type=quant_type,
            code=get_fp4_values(device),
            dtype=torch.float32
        )
    else:
        # Use the absmax from quant_state
        absmax = quant_state.absmax

    # Get the shape of the original tensor
    out_shape = quant_state.shape
    n = np.prod(out_shape)

    # Create output tensor if not provided
    if out is None:
        out = torch.empty(out_shape, device=device, dtype=quant_state.dtype)

    # Optimize unpacking using vectorized operations
    # Unpack in a single vectorized operation using advanced indexing

    # Create buffer for the unpacked indices (more efficient than multiple small operations)
    unpacked = torch.empty(n, dtype=torch.int32, device=device)

    # Calculate unpacked indices directly
    num_bytes = A.numel()

    # Process even indices (high 4 bits)
    bytes_tensor = A.to(torch.int32)  # Convert once for all operations

    # Process all at once with optimized view reshaping
    even_indices = torch.arange(0, n, 2, device=device)
    if even_indices.numel() > 0:
        high_bits = torch.bitwise_and(
            torch.bitwise_right_shift(bytes_tensor[even_indices // 2], 4),
            0xF
        )
        # Use index_put_ for efficient writing
        unpacked.index_put_((even_indices,), high_bits)

    # Process odd indices (low 4 bits)
    odd_indices = torch.arange(1, n, 2, device=device)
    odd_indices = odd_indices[odd_indices < n]  # Make sure we don't go out of bounds

    if odd_indices.numel() > 0:
        low_bits = torch.bitwise_and(bytes_tensor[odd_indices // 2], 0xF)
        unpacked.index_put_((odd_indices,), low_bits)

    # Get FP4 values and dequantize
    fp4_values = quant_state.code
    dequantized_flat = fp4_values[unpacked]

    # Reshape directly to the output shape in one step if possible
    if out_shape == dequantized_flat.shape:
        dequantized = dequantized_flat
    else:
        # Reshape to blocks for scaling with absmax
        if n // blocksize == absmax.numel():
            dequantized_blocks = dequantized_flat.view(-1, blocksize)

            # Scale by absmax using broadcasting
            dequantized_blocks = dequantized_blocks * absmax.unsqueeze(1)

            # Reshape to final shape
            dequantized = dequantized_blocks.view(out_shape)
        else:
            # Fallback for non-standard shapes
            dequantized_blocks = dequantized_flat.reshape(-1, blocksize)
            dequantized_blocks = dequantized_blocks * absmax.unsqueeze(1)
            dequantized = dequantized_blocks.reshape(out_shape)

    # Convert to correct dtype in-place if possible
    if dequantized.dtype != quant_state.dtype:
        out.copy_(dequantized)
        return out
    else:
        return dequantized
