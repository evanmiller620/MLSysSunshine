/*
 * CUDA Kernel Template for FlashInfer Competition.
 *
 * Implement your kernel logic here. The entry point function name should match
 * the `entry_point` setting in config.toml.
 *
 * See the track definition for required function signature and semantics.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void my_cuda_kernel(half* input, half* output, int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx];
    }
}
__global__ void winning_kernel(half* a, half* b, half* out, int N) {
    /*
     * Your CUDA kernel implementation.
     *
     * TODO: Implement your kernel according to the track definition.
     * The function signature should match the track requirements.
     */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = __hadd(a[idx], b[idx]); 
    }
}

extern "C" {
    
    void flashinfer_entry(void* a_ptr, void* b_ptr, void* out_ptr) {
        
        half* a = reinterpret_cast<half*>(a_ptr);
        half* b = reinterpret_cast<half*>(b_ptr);
        half* out = reinterpret_cast<half*>(out_ptr);

        int N = 1024; // Example

        dim3 block(256);
        dim3 grid((N + 255) / 256);

        winning_kernel<<<grid, block>>>(a, b, out, N);
    }
}