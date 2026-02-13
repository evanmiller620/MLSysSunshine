"""
TVM FFI Bindings Template for CUDA Kernels.

This file provides Python bindings for your CUDA kernel using TVM FFI.
The entry point function name should match the `entry_point` setting in config.toml.

See the track definition for required function signature and semantics.
"""

import ctypes
from tvm.ffi import register_func


@register_func("flashinfer.kernel")
def kernel(a_tvm, b_tvm, n):
    """
    Python binding for your CUDA kernel.

    TODO: Implement the binding according to the track definition.
    This function should:
    1. Accept the inputs as specified by the track definition
    2. Launch your CUDA kernel with appropriate grid/block dimensions
    3. Return outputs as specified by the track definition
    """
    def get_ptr(tvm_tensor):
        return ctypes.cast(tvm_tensor.data, ctypes.c_void_p)

    lib.flashinfer_entry(
        get_ptr(a_tvm), 
        get_ptr(b_tvm), 
        get_ptr(output_tvm)
    )
