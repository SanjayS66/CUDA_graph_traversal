#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include "csr.h"
#include <cuda_runtime.h>

#define CUDA_CHECK(call) /* ... */

struct DeviceCSR {    //deviceCSR differnet because it has to be stored in the GPU memory
    int V, E;
    int* row_offset;
    int* col_index;
    float* weights;
};

#ifdef __cplusplus
extern "C" {
#endif

DeviceCSR copy_csr_to_device(const CSRGraph* h_csr);    //copy CPU memory to GPU memory
void free_device_csr(DeviceCSR* d_csr);   //Free GPU memory

#ifdef __cplusplus
}
#endif

// Device-only function — not extern "C", only usable inside .cu files
__device__ float atomicMinFloat(float* addr, float value);

#endif