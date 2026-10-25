//sssp.h

#ifndef SSSP_H
#define SSSP_H

#include "csr.h"

#ifdef __cplusplus
extern "C" {
#endif

// CPU implementation of the Djikstras algo
float* bellman_ford(CSRGraph* csr, int src);

// GPU implementations using different traversal strategies.
float* bellman_GPU_tpe(const CSRGraph* csr, int src);
float* bellman_GPU_wpv(const CSRGraph* csr, int src);
float* bellman_GPU_tpv(const CSRGraph* csr, int src);

#ifdef __cplusplus
}
#endif

#endif // SSSP_H