#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "cuda_utils.h"
#include "csr.h"
#include "timer.h"
#include "sssp.h"

// ============================================================================
// CUDA Kernels for the 3 SSSP Strategies
// ============================================================================

// 1. Thread-Per-Vertex (T-P-V)
__global__ void relax_kernel_tpv(int* row_offset, int* col_idx, float* weights, float* dist, int V, int* changed_flag) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= V) return;

    for (int k = row_offset[u]; k < row_offset[u + 1]; k++) {
        int v = col_idx[k];
        float w = weights[k];
        if (dist[u] + w < dist[v]) {
            atomicMinFloat(&dist[v], dist[u] + w);
            atomicExch(changed_flag, 1);
        }
    }
}

// 2. Thread-Per-Edge (T-P-E)
__global__ void relax_kernel_tpe(int* src_idx, int* col_idx, float* weights, float* dist, int E, int* changed_flag) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;

    int u = src_idx[e];
    int v = col_idx[e];
    float w = weights[e];

    if (dist[u] + w < dist[v]) {
        atomicMinFloat(&dist[v], dist[u] + w);
        atomicExch(changed_flag, 1);
    }
}

// 3. Warp-Per-Vertex (T-P-W / W-P-V)
__global__ void relax_kernel_tpw(int* row_offset, int* col_idx, float* weights, float* dist, int V, int* changed_flag) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int u = tid / 32;
    int lane_id = tid % 32;

    if (u >= V) return;

    for (int k = row_offset[u] + lane_id; k < row_offset[u + 1]; k += 32) {
        int v = col_idx[k];
        float w = weights[k];
        if (dist[u] + w < dist[v]) {
            atomicMinFloat(&dist[v], dist[u] + w);
            atomicExch(changed_flag, 1);
        }
    }
}

// ============================================================================
// Host Algorithm Implementations
// ============================================================================

// CPU Bellman-Ford Baseline
float* run_bellman_cpu(const CSRGraph* csr, int src, int* rounds_out) {
    int V = csr->V;
    const int* row_offset = csr->row_offset;
    const int* col_idx = csr->col_index;
    const float* weights = csr->weights;

    float* dist = (float*)malloc(V * sizeof(float));
    for (int i = 0; i < V; i++) dist[i] = 1e9f;
    dist[src] = 0.0f;

    int rounds = 0;
    bool changed = true;
    while (rounds < V - 1 && changed) {
        changed = false;
        for (int u = 0; u < V; u++) {
            if (dist[u] >= 1e9f) continue;
            for (int k = row_offset[u]; k < row_offset[u + 1]; k++) {
                int v = col_idx[k];
                float w = weights[k];
                if (dist[u] + w < dist[v]) {
                    dist[v] = dist[u] + w;
                    changed = true;
                }
            }
        }
        rounds++;
    }
    if (rounds_out) *rounds_out = rounds;
    return dist;
}

// GPU Thread-Per-Vertex
float* run_bellman_gpu_tpv(const CSRGraph* csr, int src, int* rounds_out) {
    DeviceCSR d_csr = copy_csr_to_device(csr);
    int V = d_csr.V;

    float* d_dist;
    CUDA_CHECK(cudaMalloc((void**)&d_dist, V * sizeof(float)));
    int* d_changed;
    CUDA_CHECK(cudaMalloc((void**)&d_changed, sizeof(int)));

    float* h_init = (float*)malloc(V * sizeof(float));
    for (int i = 0; i < V; i++) h_init[i] = 1e9f;
    h_init[src] = 0.0f;
    CUDA_CHECK(cudaMemcpy(d_dist, h_init, V * sizeof(float), cudaMemcpyHostToDevice));
    free(h_init);

    int blockSize = 256;
    int numBlocks = (V + blockSize - 1) / blockSize;

    int rounds = 0;
    int h_changed = 1;
    while (rounds < V - 1 && h_changed) {
        h_changed = 0;
        CUDA_CHECK(cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice));
        relax_kernel_tpv<<<numBlocks, blockSize>>>(d_csr.row_offset, d_csr.col_index,
                                                   d_csr.weights, d_dist, V, d_changed);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        rounds++;
    }
    if (rounds_out) *rounds_out = rounds;

    float* dist = (float*)malloc(V * sizeof(float));
    CUDA_CHECK(cudaMemcpy(dist, d_dist, V * sizeof(float), cudaMemcpyDeviceToHost));

    free_device_csr(&d_csr);
    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_changed));
    return dist;
}

// GPU Thread-Per-Edge
float* run_bellman_gpu_tpe(const CSRGraph* csr, int src, int* rounds_out) {
    DeviceCSR d_csr = copy_csr_to_device(csr);
    int V = d_csr.V;
    int E = d_csr.E;

    // Build source index array
    int* h_src = (int*)malloc(E * sizeof(int));
    for (int u = 0; u < V; u++) {
        for (int k = csr->row_offset[u]; k < csr->row_offset[u + 1]; k++) {
            h_src[k] = u;
        }
    }
    int* d_src;
    CUDA_CHECK(cudaMalloc((void**)&d_src, E * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_src, h_src, E * sizeof(int), cudaMemcpyHostToDevice));
    free(h_src);

    float* d_dist;
    CUDA_CHECK(cudaMalloc((void**)&d_dist, V * sizeof(float)));
    int* d_changed;
    CUDA_CHECK(cudaMalloc((void**)&d_changed, sizeof(int)));

    float* h_init = (float*)malloc(V * sizeof(float));
    for (int i = 0; i < V; i++) h_init[i] = 1e9f;
    h_init[src] = 0.0f;
    CUDA_CHECK(cudaMemcpy(d_dist, h_init, V * sizeof(float), cudaMemcpyHostToDevice));
    free(h_init);

    int blockSize = 256;
    int numBlocks = (E + blockSize - 1) / blockSize;

    int rounds = 0;
    int h_changed = 1;
    while (rounds < V - 1 && h_changed) {
        h_changed = 0;
        CUDA_CHECK(cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice));
        relax_kernel_tpe<<<numBlocks, blockSize>>>(d_src, d_csr.col_index,
                                                   d_csr.weights, d_dist, E, d_changed);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        rounds++;
    }
    if (rounds_out) *rounds_out = rounds;

    float* dist = (float*)malloc(V * sizeof(float));
    CUDA_CHECK(cudaMemcpy(dist, d_dist, V * sizeof(float), cudaMemcpyDeviceToHost));

    free_device_csr(&d_csr);
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_changed));
    return dist;
}

// GPU Warp-Per-Vertex
float* run_bellman_gpu_tpw(const CSRGraph* csr, int src, int* rounds_out) {
    DeviceCSR d_csr = copy_csr_to_device(csr);
    int V = d_csr.V;

    float* d_dist;
    CUDA_CHECK(cudaMalloc((void**)&d_dist, V * sizeof(float)));
    int* d_changed;
    CUDA_CHECK(cudaMalloc((void**)&d_changed, sizeof(int)));

    float* h_init = (float*)malloc(V * sizeof(float));
    for (int i = 0; i < V; i++) h_init[i] = 1e9f;
    h_init[src] = 0.0f;
    CUDA_CHECK(cudaMemcpy(d_dist, h_init, V * sizeof(float), cudaMemcpyHostToDevice));
    free(h_init);

    int blockSize = 256;
    int warpsPerBlock = blockSize / 32;
    int numBlocks = (V + warpsPerBlock - 1) / warpsPerBlock;

    int rounds = 0;
    int h_changed = 1;
    while (rounds < V - 1 && h_changed) {
        h_changed = 0;
        CUDA_CHECK(cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice));
        relax_kernel_tpw<<<numBlocks, blockSize>>>(d_csr.row_offset, d_csr.col_index,
                                                   d_csr.weights, d_dist, V, d_changed);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        rounds++;
    }
    if (rounds_out) *rounds_out = rounds;

    float* dist = (float*)malloc(V * sizeof(float));
    CUDA_CHECK(cudaMemcpy(dist, d_dist, V * sizeof(float), cudaMemcpyDeviceToHost));

    free_device_csr(&d_csr);
    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_changed));
    return dist;
}

// ============================================================================
// Verification Helper
// ============================================================================
int verify_distances(const float* baseline, const float* test_arr, int V, float epsilon = 1e-3f) {
    int mismatches = 0;
    int max_report = 5;
    for (int i = 0; i < V; i++) {
        if (fabs(baseline[i] - test_arr[i]) >= epsilon) {
            mismatches++;
            if (mismatches <= max_report) {
                printf("    [Mismatch @ %d] Baseline=%.4f vs Test=%.4f (diff=%.4f)\n",
                       i, baseline[i], test_arr[i], fabs(baseline[i] - test_arr[i]));
            }
        }
    }
    return mismatches;
}

// ============================================================================
// Main Benchmark Runner
// ============================================================================
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <graph_file> [--undirected] [--dump] [--src <int>]\n", argv[0]);
        return 1;
    }

    const char* input_path = argv[1];
    int directed = 1;
    int dump = 0;
    int source_vertex = 0;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--undirected") == 0) {
            directed = 0;
        } else if (strcmp(argv[i], "--dump") == 0) {
            dump = 1;
        } else if (strcmp(argv[i], "--src") == 0 && i + 1 < argc) {
            source_vertex = atoi(argv[++i]);
        }
    }

    printf("\n==================================================================\n");
    printf("         SSSP LOAD-BALANCING BENCHMARK & VERIFICATION\n");
    printf("==================================================================\n");
    printf("Graph File   : %s\n", input_path);
    printf("Directed     : %s\n", directed ? "Yes" : "No (Edges doubled)");
    printf("Source Vertex: %d\n", source_vertex);

    // Warm up CUDA runtime to avoid driver init penalty in timings
    CUDA_CHECK(cudaFree(0));

    // 1. Read graph
    Timer t_read, t_csr;
    timer_start(&t_read);
    EdgeListGraph g = read_graph_from_file(input_path);
    timer_stop(&t_read);

    // 2. Build CSR
    timer_start(&t_csr);
    CSRGraph csr = build_csr(&g, directed);
    timer_stop(&t_csr);

    if (!validate_csr(&csr)) {
        fprintf(stderr, "Error: CSR validation failed!\n");
        free(g.edges);
        free_csr(&csr);
        return 1;
    }

    printf("Graph Specs  : V = %d, E (CSR) = %d\n", csr.V, csr.E);
    printf("I/O Timings  : Read = %.2f ms, Build CSR = %.2f ms\n",
           timer_elapsed_ms(&t_read), timer_elapsed_ms(&t_csr));

    if (dump) {
        dump_csr(&csr);
    }

    printf("\nRunning implementations...\n");

    Timer t_timer;
    int cpu_rounds = 0, tpv_rounds = 0, tpe_rounds = 0, tpw_rounds = 0;

    // A. CPU Baseline
    printf("  [1/4] Running CPU Bellman-Ford... ");
    fflush(stdout);
    timer_start(&t_timer);
    float* dist_cpu = run_bellman_cpu(&csr, source_vertex, &cpu_rounds);
    timer_stop(&t_timer);
    double cpu_time = timer_elapsed_ms(&t_timer);
    printf("Done (%.2f ms, %d rounds)\n", cpu_time, cpu_rounds);

    // B. GPU Thread-Per-Vertex (T-P-V)
    printf("  [2/4] Running GPU Thread-Per-Vertex (TPV)... ");
    fflush(stdout);
    timer_start(&t_timer);
    float* dist_tpv = run_bellman_gpu_tpv(&csr, source_vertex, &tpv_rounds);
    timer_stop(&t_timer);
    double tpv_time = timer_elapsed_ms(&t_timer);
    printf("Done (%.2f ms, %d rounds)\n", tpv_time, tpv_rounds);

    // C. GPU Thread-Per-Edge (T-P-E)
    printf("  [3/4] Running GPU Thread-Per-Edge (TPE)... ");
    fflush(stdout);
    timer_start(&t_timer);
    float* dist_tpe = run_bellman_gpu_tpe(&csr, source_vertex, &tpe_rounds);
    timer_stop(&t_timer);
    double tpe_time = timer_elapsed_ms(&t_timer);
    printf("Done (%.2f ms, %d rounds)\n", tpe_time, tpe_rounds);

    // D. GPU Warp-Per-Vertex (T-P-W)
    printf("  [4/4] Running GPU Warp-Per-Vertex (TPW)... ");
    fflush(stdout);
    timer_start(&t_timer);
    float* dist_tpw = run_bellman_gpu_tpw(&csr, source_vertex, &tpw_rounds);
    timer_stop(&t_timer);
    double tpw_time = timer_elapsed_ms(&t_timer);
    printf("Done (%.2f ms, %d rounds)\n", tpw_time, tpw_rounds);

    // ========================================================================
    // Correctness Verification
    // ========================================================================
    printf("\nVerifying Correctness Against CPU Baseline:\n");
    int tpv_mismatches = verify_distances(dist_cpu, dist_tpv, csr.V);
    int tpe_mismatches = verify_distances(dist_cpu, dist_tpe, csr.V);
    int tpw_mismatches = verify_distances(dist_cpu, dist_tpw, csr.V);

    // ========================================================================
    // Results & Speedup Comparison Table
    // ========================================================================
    printf("\n====================================================================================================\n");
    printf("%-30s | %-8s | %-12s | %-14s | %-16s\n",
           "Strategy", "Rounds", "Time (ms)", "Speedup (vs CPU)", "Verification");
    printf("----------------------------------------------------------------------------------------------------\n");
    
    printf("%-30s | %-8d | %-12.4f | %-14s | %-16s\n",
           "1. CPU Bellman-Ford", cpu_rounds, cpu_time, "1.00x (Base)", "BASELINE");

    char tpv_verif[32], tpe_verif[32], tpw_verif[32];
    snprintf(tpv_verif, sizeof(tpv_verif), "%s", (tpv_mismatches == 0) ? "PASSED (100%)" : "FAILED");
    snprintf(tpe_verif, sizeof(tpe_verif), "%s", (tpe_mismatches == 0) ? "PASSED (100%)" : "FAILED");
    snprintf(tpw_verif, sizeof(tpw_verif), "%s", (tpw_mismatches == 0) ? "PASSED (100%)" : "FAILED");

    char tpv_speedup[16], tpe_speedup[16], tpw_speedup[16];
    snprintf(tpv_speedup, sizeof(tpv_speedup), "%.2fx", cpu_time / tpv_time);
    snprintf(tpe_speedup, sizeof(tpe_speedup), "%.2fx", cpu_time / tpe_time);
    snprintf(tpw_speedup, sizeof(tpw_speedup), "%.2fx", cpu_time / tpw_time);

    printf("%-30s | %-8d | %-12.4f | %-16s | %-16s\n",
           "2. GPU Thread-Per-Vertex", tpv_rounds, tpv_time, tpv_speedup, tpv_verif);

    printf("%-30s | %-8d | %-12.4f | %-16s | %-16s\n",
           "3. GPU Thread-Per-Edge", tpe_rounds, tpe_time, tpe_speedup, tpe_verif);

    printf("%-30s | %-8d | %-12.4f | %-16s | %-16s\n",
           "4. GPU Warp-Per-Vertex", tpw_rounds, tpw_time, tpw_speedup, tpw_verif);

    printf("====================================================================================================\n\n");

    // Clean up
    free(g.edges);
    free_csr(&csr);
    free(dist_cpu);
    free(dist_tpv);
    free(dist_tpe);
    free(dist_tpw);

    return (tpv_mismatches == 0 && tpe_mismatches == 0 && tpw_mismatches == 0) ? 0 : 1;
}
