#include "cuda_utils.h"

DeviceCSR copy_csr_to_device(const CSRGraph* h_csr){
    //take the host CSR and copy into the GPU memory
    DeviceCSR d_csr;
    d_csr.V = h_csr->V;
    d_csr.E = h_csr->E;

    size_t row_offset_bytes = (h_csr->V + 1) * sizeof(int);
    size_t col_idx_bytes = (h_csr->E) * sizeof(int);
    size_t weights_bytes = (h_csr->E) * sizeof(float);

    // just allocate the required memory
    CUDA_CHECK(cudaMalloc(&d_csr.row_offset, row_offset_bytes));
    CUDA_CHECK(cudaMalloc(&d_csr.col_index,  col_idx_bytes));
    CUDA_CHECK(cudaMalloc(&d_csr.weights,    weights_bytes));

    // Copy actual array contents from host to device
    CUDA_CHECK(cudaMemcpy(d_csr.row_offset, h_csr->row_offset,
                           row_offset_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csr.col_index, h_csr->col_index,
                           col_idx_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csr.weights, h_csr->weights,
                           weights_bytes, cudaMemcpyHostToDevice));

    return d_csr;
}

void free_device_csr(DeviceCSR* d_csr){
    // release each device allocation made in copy_csr_to_device
    cudaFree(d_csr->row_offset);
    cudaFree(d_csr->col_index);
    cudaFree(d_csr->weights);
}

__device__ float atomicMinFloat(float* addr, float value){
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int;
    int assumed;

    // Only try to update if our candidate is actually smaller than
    // whatever is currently stored (bit-reinterpreted back to float).
    while (value < __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(value));
        // If atomicCAS's return equals what we assumed, our write succeeded.
        // If not, another thread updated it in between — loop and retry
        // against the newer value.
        if (assumed == old) break;
    }

    return __int_as_float(old);
}