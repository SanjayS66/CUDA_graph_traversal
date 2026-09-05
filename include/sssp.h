//sssp.h

#ifndef SSSP_H
#define SSSP_H

#include "csr.h"

#ifdef __cplusplus
extern "C" {
#endif

// CPU implementation of the Djikstras algo
float* bellman_ford(CSRGraph* csr, int src);

// GPU counterpart — same contract as djikstras(), implemented in sssp_gpu.cu
float* bellman_GPU(CSRGraph* csr, int src);

#ifdef __cplusplus
}
#endif

#endif // SSSP_H