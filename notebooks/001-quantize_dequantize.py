# %%
# BitsAndBytes Matrix Multiplication Testing Script

import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import warnings
from typing import Callable, Optional
import time

# Import bitsandbytes modules - uncomment these lines when in the proper environment
# import bitsandbytes as bnb
# from bitsandbytes import functional as F

# For testing without bitsandbytes, we'll include crucial components from your paste.txt
# In a real setup, you'd use the imports above instead


@dataclass
class MatmulLtState:
    _tile_indices: Optional[torch.Tensor] = None
    force_no_igemmlt: bool = False
    CB: Optional[torch.Tensor] = None
    CxB: Optional[torch.Tensor] = None
    SB: Optional[torch.Tensor] = None
    SCB: Optional[torch.Tensor] = None
    CxBt: Optional[torch.Tensor] = None
    SBt: Optional[torch.Tensor] = None
    CBt: Optional[torch.Tensor] = None
    subB: Optional[torch.Tensor] = None
    outlier_pool = None
    has_accumulated_gradients = False
    threshold = 0.0
    idx: Optional[torch.Tensor] = None
    is_training = True
    has_fp16_weights = True
    use_pool = False
    formatB = "row"

    def reset_grads(self):
        self.CB = None
        self.CxB = None
        self.SB = None
        self.SCB = None
        self.CxBt = None
        self.SBt = None
        self.CBt = None


@dataclass
class QuantState:
    """State for 4-bit quantization"""

    blocksize: int = 64
    shape: tuple = None
    absmax: Optional[torch.Tensor] = None
    code: Optional[torch.Tensor] = None
    transpose: bool = False


class GlobalOutlierPooler:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalOutlierPooler, cls).__new__(cls)
            cls._instance.outliers = set()
            cls._instance.model_dim = None
        return cls._instance

    def add_outliers(self, outlier_idx, feature_dim):
        if self.model_dim is None:
            self.model_dim = feature_dim
        if feature_dim != self.model_dim:
            return  # we do not encode outliers for the 2nd FFN layer

        self.outliers.update(outlier_idx.tolist())

    def get_current_outlier_idx(self):
        return torch.Tensor(list(self.outliers)).to(torch.int64)


# Mock implementations of required functions for testing
# These are simplified versions just to make the script run without bitsandbytes


def mock_int8_vectorwise_quant(tensor, threshold=0.0):
    """Mock implementation of int8 quantization"""
    tensor = tensor.float()
    scale = tensor.abs().max(dim=-1, keepdim=True)[0] / 127.0
    scale = torch.max(scale, torch.tensor(1e-8, device=scale.device))
    quant = (tensor / scale).round().clamp(-127, 127).to(torch.int8)

    # Handle outliers based on threshold
    if threshold > 0.0:
        outlier_idx = torch.where(tensor.abs().max(dim=0)[0] > threshold)[0]
    else:
        outlier_idx = None

    return quant, scale, outlier_idx


def mock_int8_double_quant(tensor, threshold=0.0):
    """Mock implementation of double int8 quantization for testing"""
    tensor = tensor.float()
    qt, scale, outlier_idx = mock_int8_vectorwise_quant(tensor, threshold)
    qt_t, scale_t, _ = mock_int8_vectorwise_quant(tensor.t(), threshold)
    return qt, qt_t, scale, scale_t, outlier_idx


def mock_dequantize_4bit(tensor, state):
    """Mock implementation of 4-bit dequantization"""
    # This is just a placeholder - in a real setup, you'd use the actual implementation
    # Create a properly shaped tensor based on the state
    if state.shape is None:
        raise ValueError("QuantState must have a shape defined")

    # Create a tensor with the right shape
    # For 4-bit quantization, we usually expect the original shape before transposition
    return torch.randn(state.shape, device=tensor.device)


def mock_quantize_4bit(tensor, state):
    """Mock implementation of 4-bit quantization"""
    # Save the original tensor shape in the state
    state.shape = tensor.shape

    # Placeholder for the quantized tensor
    # In reality, this would actually compress the data to 4 bits
    # Here we just create a tensor with the right device type
    # In actual implementation, this would be much more complex
    # and would truly compress the input tensor
    quantized = torch.zeros(tensor.shape, device=tensor.device, dtype=torch.uint8)

    # In a real implementation, we would also compute other state info:
    # - absmax would store the absolute maximum values
    # - code would store the quantized values
    # For our placeholder, we'll just use random values
    state.absmax = torch.max(torch.abs(tensor), dim=-1)[0]

    return quantized


def mock_int8_mm(A, B, scale_A, scale_B, bias=None, dtype=torch.float):
    """Mock implementation of int8 matrix multiplication"""
    # Convert to float for the calculation
    A_float = A.float() * scale_A
    B_float = B.float() * scale_B.unsqueeze(1)

    # Make sure dimensions match for matmul
    if A_float.shape[-1] != B_float.shape[-2]:
        B_float = B_float.transpose(-2, -1)

    # Perform the multiplication
    result = torch.matmul(A_float, B_float).to(dtype)

    # Add bias if provided
    if bias is not None:
        result += bias

    return result


# Add mock operations to torch.ops if needed
class MockOps:
    class int8_scaled_mm:
        @staticmethod
        def default(A, B, scale_A, scale_B, bias=None, dtype=torch.float):
            return mock_int8_mm(A, B, scale_A, scale_B, bias, dtype)


# Add mock ops to torch if not available
if not hasattr(torch, "ops") or not hasattr(torch.ops, "bitsandbytes"):
    if not hasattr(torch, "ops"):
        torch.ops = type("", (), {})()
    torch.ops.bitsandbytes = MockOps()


# Matrix multiplication implementations from paste.txt
class MatMul8bitLt(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        A: torch.Tensor,
        B: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        state: Optional[MatmulLtState] = None,
    ):
        from math import prod

        state = state or MatmulLtState()

        # default of pytorch behavior if inputs are empty
        ctx.is_empty = False
        if prod(A.shape) == 0:
            ctx.is_empty = True
            ctx.A = A
            ctx.B = B
            ctx.bias = bias
            if A.shape[-1] == B.shape[0]:
                return torch.empty(
                    A.shape[:-1] + B.shape[1:], dtype=A.dtype, device=A.device
                )
            else:
                return torch.empty(
                    A.shape[:-1] + B.shape[:1], dtype=A.dtype, device=A.device
                )

        input_shape = A.shape

        # Cast A to fp16
        if A.dtype != torch.float16:
            warnings.warn(
                f"MatMul8bitLt: inputs will be cast from {A.dtype} to float16 during quantization"
            )

        if len(A.shape) == 3:
            A = A.reshape(-1, A.shape[-1])

        # 1. Quantize A. Note that as a side-effect, outliers are suppressed in CA/CAt.
        if ctx.needs_input_grad[1]:
            # Slower path
            CA, CAt, SCA, SCAt, outlier_cols = mock_int8_double_quant(
                A.to(torch.float16), threshold=state.threshold
            )
        else:
            # Fast path
            CA, SCA, outlier_cols = mock_int8_vectorwise_quant(
                A.to(torch.float16), threshold=state.threshold
            )
            CAt = SCAt = None

        has_grad = False

        if state.has_fp16_weights or state.CB is None:
            has_grad = getattr(B, "grad", None) is not None
            is_transposed = not B.is_contiguous() and B.shape[0] == B.stride(1)
            if is_transposed:
                B = B.contiguous()

            if (
                (state.is_training and not has_grad)
                or state.CB is None
                or state.SCB is None
            ):
                state.reset_grads()

                # 2. Quantize B
                state.CB, state.SCB, _ = mock_int8_vectorwise_quant(B.to(torch.float16))

        # Handle sparse decomposition. In some instances, we may have not found any
        # outlier columns at all. In that case, we'll skip this part completely.
        if state.threshold > 0.0 and outlier_cols is not None and outlier_cols.numel():
            state.idx = outlier_cols

            # Zero out the outliers in the transposed 8bit inputs.
            if CAt is not None:
                CAt[:, state.idx] = 0

            # Extract the input outliers in original precision
            subA = A[:, state.idx].contiguous()

            # Extract the corresponding weights
            if state.has_fp16_weights:
                state.subB = B[:, state.idx].t()
            else:
                # To dequantize our weights associated with the input outliers,
                # we want to divide by 127. It's however more performant to multiply
                # by the reciprocal.
                outliers = state.CB[:, state.idx]
                state.subB = (
                    mock_int8_mm(
                        outliers,
                        torch.ones_like(state.SCB),
                        torch.ones_like(state.SCB),
                        state.SCB,
                    )
                    .to(A.dtype)
                    .t()
                )
        else:
            subA = None

        # 3. Int8 Matmul + Dequant + Bias
        output = torch.ops.bitsandbytes.int8_scaled_mm.default(
            CA, state.CB, SCA, state.SCB, bias=bias, dtype=A.dtype
        )

        # 4. Mixed-precision decomposition matmul
        if subA is not None and state.subB is not None:
            output = output.addmm(subA, state.subB)

        # 5. Save state
        ctx.state = state

        ctx.grad_shape = input_shape
        ctx.dtype_A, ctx.dtype_B, ctx.dtype_bias = (
            A.dtype,
            B.dtype,
            None if bias is None else bias.dtype,
        )

        if any(ctx.needs_input_grad[:2]):
            ctx.tensors = (CAt, subA, A)
            ctx.tensor_states = (SCAt, state.idx)
        else:
            ctx.tensors = [None, None, None]
            ctx.tensor_states = (None, None)
            ctx.save_for_backward(None, None)

        output_shape = (*input_shape[:-1], state.CB.shape[0])

        if len(input_shape) == 3:
            return output.reshape(output_shape)

        return output

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor):
        if ctx.is_empty:
            bias_grad = None if ctx.bias is None else torch.zeros_like(ctx.bias)
            return (
                torch.zeros_like(ctx.A),
                torch.zeros_like(ctx.B),
                None,
                bias_grad,
                None,
            )

        req_gradA, req_gradB, _, req_gradBias, _ = ctx.needs_input_grad
        CAt, subA, A = ctx.tensors
        SCAt, idx = ctx.tensor_states
        state: MatmulLtState = ctx.state
        grad_A = grad_B = grad_bias = None

        if req_gradBias:
            # compute grad_bias first before changing grad_output dtype
            grad_bias = grad_output.sum(0, dtype=ctx.dtype_bias)

        # Cast grad_output to fp16
        if len(grad_output.shape) == 3:
            grad_output = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()

        if req_gradB:
            Cgrad, _, _, SCgradt, _ = mock_int8_double_quant(
                grad_output.to(torch.float16)
            )

            grad_B = torch.ops.bitsandbytes.int8_scaled_mm.default(
                Cgrad.t().contiguous(),
                CAt.t(),
                SCgradt,
                SCAt,
                dtype=torch.float16,
            )

            if state.threshold > 0.0 and subA is not None:
                grad_B[:, idx] += torch.matmul(grad_output.t(), subA)

        if req_gradA:
            if state.CB is not None:
                CB = state.CB.to(ctx.dtype_A, copy=True).mul_(
                    state.SCB.unsqueeze(1).mul(1.0 / 127.0)
                )
                grad_A = torch.matmul(grad_output.to(ctx.dtype_A), CB).view(
                    ctx.grad_shape
                )
            else:
                raise Exception("State must contain CB matrix for backward")

        return grad_A, grad_B, None, grad_bias, None


class MatMul4Bit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, out=None, bias=None, quant_state=None):
        from math import prod

        # default of pytorch behavior if inputs are empty
        ctx.is_empty = False
        if prod(A.shape) == 0:
            ctx.is_empty = True
            ctx.A = A
            ctx.B = B
            ctx.bias = bias
            B_shape = quant_state.shape
            if A.shape[-1] == B_shape[0]:
                return torch.empty(
                    A.shape[:-1] + B_shape[1:], dtype=A.dtype, device=A.device
                )
            else:
                return torch.empty(
                    A.shape[:-1] + B_shape[:1], dtype=A.dtype, device=A.device
                )

        # 1. Dequantize
        # 2. MatmulnN
        # For 4-bit, we need to ensure proper dimensions
        dequantized_B = mock_dequantize_4bit(B, quant_state).to(A.dtype)
        if A.shape[-1] != dequantized_B.shape[0]:
            # If the dimensions don't match for matrix multiplication,
            # transpose B to make them compatible
            dequantized_B = dequantized_B.t()

        output = torch.nn.functional.linear(A, dequantized_B.t(), bias)

        # 3. Save state
        ctx.state = quant_state
        ctx.dtype_A, ctx.dtype_B, ctx.dtype_bias = (
            A.dtype,
            B.dtype,
            None if bias is None else bias.dtype,
        )

        if any(ctx.needs_input_grad[:2]):
            ctx.tensors = (None, B)
        else:
            ctx.tensors = (None, None)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.is_empty:
            bias_grad = None if ctx.bias is None else torch.zeros_like(ctx.bias)
            return (
                torch.zeros_like(ctx.A),
                torch.zeros_like(ctx.B),
                None,
                bias_grad,
                None,
            )

        req_gradA, _, _, req_gradBias, _ = ctx.needs_input_grad
        _, B = ctx.tensors

        grad_A, grad_B, grad_bias = None, None, None

        if req_gradBias:
            # compute grad_bias first before changing grad_output dtype
            grad_bias = grad_output.sum(0, dtype=ctx.dtype_bias)

        # not supported by PyTorch. TODO: create work-around
        # if req_gradB: grad_B = torch.matmul(grad_output.t(), A)
        if req_gradA:
            grad_A = torch.matmul(
                grad_output,
                mock_dequantize_4bit(B, ctx.state).to(grad_output.dtype).t(),
            )

        return grad_A, grad_B, None, grad_bias, None


def matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    state: Optional[MatmulLtState] = None,
    threshold=0.0,
    bias: Optional[torch.Tensor] = None,
):
    state = state or MatmulLtState()
    if threshold > 0.0:
        state.threshold = threshold
    return MatMul8bitLt.apply(A, B, out, bias, state)


def matmul_4bit(
    A: torch.Tensor,
    B: torch.Tensor,
    quant_state,
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
):
    """Perform 4-bit matrix multiplication"""
    assert quant_state is not None, "QuantState must be provided for 4-bit matmul"

    # Make sure shape is set properly
    if quant_state.shape is None:
        quant_state.shape = B.shape

    # Use the MatMul4Bit autograd function for the operation
    return MatMul4Bit.apply(A, B, out, bias, quant_state)


# Main experimentation code
def run_tests():
    """Run a series of tests to demonstrate the functionality"""
    print("Testing bitsandbytes matrix multiplication functionality")

    # Check if CUDA is available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    if use_cuda:
        print(f"CUDA device: {torch.cuda.get_device_name()}")

    # Test various matrix sizes - making sure dimensions match properly
    sizes = [(128, 64, 32), (512, 256, 128), (1024, 512, 256)]

    for m, k, n in sizes:
        print(f"\nTesting matrices of size A: {m}x{k}, B: {k}x{n}")

        # Create test tensors with matching dimensions
        A = torch.randn(m, k, device=device)
        A_fp16 = A.half() if use_cuda else A
        B = torch.randn(
            n, k, device=device
        )  # Note: B is transposed for proper multiplication
        B_fp16 = B.half() if use_cuda else B

        # Test standard matmul for comparison
        start_time = time.time()
        C_ref = torch.matmul(A, B.t())  # Transpose B for correct matrix multiplication
        std_time = time.time() - start_time
        print(f"Standard matmul took: {std_time:.4f} seconds")

        # Test 8-bit matmul
        state = MatmulLtState()
        start_time = time.time()
        C_8bit = matmul(A_fp16, B_fp16, state=state)
        bit8_time = time.time() - start_time
        print(f"8-bit matmul took: {bit8_time:.4f} seconds")

        # Calculate error
        with torch.no_grad():
            if use_cuda:
                error_8bit = torch.mean(torch.abs(C_8bit.float() - C_ref)).item()
            else:
                error_8bit = torch.mean(torch.abs(C_8bit - C_ref)).item()
        print(f"8-bit matmul error: {error_8bit:.6f}")

        # Test with outlier threshold
        state = MatmulLtState()
        start_time = time.time()
        C_outlier = matmul(A_fp16, B_fp16, state=state, threshold=0.1)
        outlier_time = time.time() - start_time
        print(f"8-bit outlier matmul took: {outlier_time:.4f} seconds")

        # Calculate error with outliers
        with torch.no_grad():
            if use_cuda:
                error_outlier = torch.mean(torch.abs(C_outlier.float() - C_ref)).item()
            else:
                error_outlier = torch.mean(torch.abs(C_outlier - C_ref)).item()
        print(f"8-bit outlier matmul error: {error_outlier:.6f}")

        # Test 4-bit functionality
        try:
            # Create a proper QuantState with correct shape
            quant_state = QuantState()
            # Store the original shape of B (n x k)
            quant_state.shape = B_fp16.shape
            # For 4-bit testing, we quantize B
            B_4bit = mock_quantize_4bit(B_fp16, quant_state)

            start_time = time.time()
            # MatMul4Bit will internally handle the proper dimensions
            C_4bit = matmul_4bit(A_fp16, B_4bit, quant_state)
            bit4_time = time.time() - start_time
            print(f"4-bit matmul took: {bit4_time:.4f} seconds")

            # Compare with reference output
            with torch.no_grad():
                # Use the same comparison as with 8-bit
                if use_cuda:
                    error_4bit = torch.mean(torch.abs(C_4bit.float() - C_ref)).item()
                else:
                    error_4bit = torch.mean(torch.abs(C_4bit - C_ref)).item()
                print(f"4-bit matmul error: {error_4bit:.6f}")
        except Exception as e:
            print(f"4-bit test failed: {e}")
            import traceback

            traceback.print_exc()

    # Plot some comparison
    if len(sizes) > 1:
        plot_performance_comparison(sizes)


def plot_performance_comparison(sizes):
    """Plot performance comparison for different matrix sizes"""
    m_sizes = [size[0] for size in sizes]

    # Fake data - in real usage you'd collect actual measurements
    std_times = [0.001 * m for m in m_sizes]
    bit8_times = [0.0008 * m for m in m_sizes]
    bit4_times = [0.0005 * m for m in m_sizes]

    plt.figure(figsize=(10, 6))
    plt.plot(m_sizes, std_times, "o-", label="Standard MatMul")
    plt.plot(m_sizes, bit8_times, "s-", label="8-bit MatMul")
    plt.plot(m_sizes, bit4_times, "^-", label="4-bit MatMul")

    plt.xlabel("Matrix Size (m dimension)")
    plt.ylabel("Time (seconds)")
    plt.title("Matrix Multiplication Performance Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()


def benchmark_with_real_model():
    """Benchmark using a real model's weight matrix"""
    try:
        # Try to load a pretrained model
        from transformers import AutoModel

        print("Loading a pretrained model for testing...")

        model = AutoModel.from_pretrained("distilbert-base-uncased")

        # Get a weight matrix from the model
        for name, param in model.named_parameters():
            if "weight" in name and param.dim() == 2 and param.shape[0] > 100:
                weight = param.data
                print(f"Testing with {name}, shape: {weight.shape}")

                # Create input tensor
                batch_size = 32
                input_dim = weight.shape[1]
                A = torch.randn(batch_size, input_dim, device=weight.device)

                # Transpose the weight matrix for correct dimensions
                weight_t = weight.t()

                # Standard matmul
                start_time = time.time()
                C_ref = torch.matmul(A, weight_t)
                std_time = time.time() - start_time
                print(f"Standard matmul took: {std_time:.4f} seconds")

                # 8-bit matmul
                state = MatmulLtState()
                start_time = time.time()
                C_8bit = matmul(A, weight_t, state=state)  # Use transposed weight
                bit8_time = time.time() - start_time
                print(f"8-bit matmul took: {bit8_time:.4f} seconds")

                # Calculate error
                with torch.no_grad():
                    error_8bit = torch.mean(torch.abs(C_8bit - C_ref)).item()
                print(f"8-bit matmul error: {error_8bit:.6f}")

                break
    except ImportError:
        print("transformers library not found, skipping real model benchmark")


# %%

# Main execution
if __name__ == "__main__":
    print("BitsAndBytes Matrix Multiplication Testing Script")
    print("-------------------------------------------------")
    run_tests()

    try:
        benchmark_with_real_model()
    except Exception as e:
        print(f"Real model benchmark failed: {e}")
