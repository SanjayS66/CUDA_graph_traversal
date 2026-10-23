#include <cstdio>
#include <cstdlib> 
#include <cstring> 
#include "cuda_utils.h"
#include "csr.h"
#include "timer.h"
#include "sssp.h"

Timer t_bellman,t_edge,t_csr;

EdgeListGraph return_edgelist(char* input_path){
    EdgeListGraph g;
    g = read_graph_from_file(input_path);
    
    printf("Read graph: V=%d E=%d from %s\n", g.V, g.E, input_path);
    return g;
}

CSRGraph return_csr(EdgeListGraph* e,int directed){
    CSRGraph csr;
    csr = build_csr(e, directed);
    if (!validate_csr(&csr)) {
        fprintf(stderr, "CSR validation FAILED\n");
        free(e->edges);
        free_csr(&csr);
        CSRGraph empty_csr = {0, 0, NULL, NULL, NULL};
        return empty_csr;
    }
    return csr;
}

__global__ void relax_kernel(int* src_idx, int* col_idx, float *weights, float* dist, int E, int* changed_flag){
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) {
        return;
    } 

    int u = src_idx[e];
    int v = col_idx[e];
    float w = weights[e];

    if (dist[u] + w < dist[v]) {
        atomicMinFloat(&dist[v], dist[u] + w);
        atomicExch(changed_flag, 1);
    }
}


__host__ float* bellman_GPU_tpe(CSRGraph* csr, int src){
    DeviceCSR d_csr = copy_csr_to_device(csr);

    int V = d_csr.V;
    int E = d_csr.E;

    // 1. Build and copy src_idx array for edges (maps edge e -> source vertex u)
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
    size_t size = V * sizeof(float);
    CUDA_CHECK(cudaMalloc((void**)&d_dist, size));

    int* d_changed;
    CUDA_CHECK(cudaMalloc((void**)&d_changed, sizeof(int)));

    // Initialize dist array: src = 0, all others = 1e9
    float* h_init = (float*)malloc(V * sizeof(float));
    for (int i = 0; i < V; i++) {
        h_init[i] = 1e9f;
    }
    h_init[src] = 0.0f;
    CUDA_CHECK(cudaMemcpy(d_dist, h_init, V * sizeof(float), cudaMemcpyHostToDevice));
    free(h_init);

    // Launch configuration sized to E
    int blockSize = 256;
    int numBlocks = (E + blockSize - 1) / blockSize;

    int i = 0;
    int h_changed = 1;
    while (i < V - 1 && h_changed) {
        h_changed = 0;
        CUDA_CHECK(cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice));
        
        relax_kernel<<<numBlocks, blockSize>>>(d_src, d_csr.col_index,
                                               d_csr.weights, d_dist, E, d_changed);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        i++;
    }
    printf("Bellman-Ford GPU (TPE) converged in %d rounds\n", i);

    float* dist = (float*)malloc(V * sizeof(float));
    CUDA_CHECK(cudaMemcpy(dist, d_dist, V * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Cleanup
    free_device_csr(&d_csr);
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_changed));
    return dist;
}


//main fnc
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input_file> [--undirected] [--dump]\n", argv[0]);
        return 1;
    }

    char *input_path = argv[1];
    int directed = 1;
    int dump = 0; // for user to select if tehy want to log all the edgse or not

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--undirected") == 0) {
            directed = 0;
        } else if (strcmp(argv[i], "--dump") == 0) {
            dump = 1;
        } else {
            fprintf(stderr, "Unrecognized argument: %s\n", argv[i]);
            return 1;
        }
    }

    timer_start(&t_edge);
    EdgeListGraph e = return_edgelist(input_path);
    timer_stop(&t_edge);
    printf("Graph Vertices (V): %d, Edges (E): %d\n", e.V, e.E);

    //debgu print
    // printf(" Edge List :\n");
    //DEBUG PRINT
    // for (int i = 0; i < e.E; i++) {
    //     printf("Edge %d: u = %d, v = %d, weight = %.3f\n", 
    //         i, 
    //         e.edges[i].u, 
    //         e.edges[i].v, 
    //         e.edges[i].w);
    // }

//  returns validated csr
    timer_start(&t_csr);
    CSRGraph csr = return_csr(&e,directed);
    timer_stop(&t_csr);
    // float* weights = csr.weights;
    // int* col_idx = csr.col_index;
    // int* row_offsets = csr.row_offset;
    // printf("Row_Offset:\n");
    // for(int i=0;i<=csr.V;i++){
    //     printf("%d    ",row_offsets[i]);
    // }printf("\n");

    // printf("Column Index:\n");
    // for(int i=0;i<csr.E;i++){
    //     printf("%d   ",col_idx[i]);
    // }
    // printf("\n");

    // printf("Weight:\n");
    // for(int i=0;i<csr.E;i++){
    //     printf("%f  ",weights[i]);
    // }
    // printf("\n");

    

    printf("CSR built and validated OK\n");
//  debug print
    // printf("  directed mode : %s\n", directed ? "directed" : "undirected (edges doubled)");
    // printf("  V             : %d\n", csr.V);
    // printf("  E (CSR)       : %d\n", csr.E);

    // if (csr.V > 0) {
    //     int min_deg = csr.row_offset[1] - csr.row_offset[0];
    //     int max_deg = min_deg;
    //     for (int i = 1; i < csr.V; i++) {
    //         int d = csr.row_offset[i + 1] - csr.row_offset[i];
    //         if (d < min_deg) min_deg = d;
    //         if (d > max_deg) max_deg = d;
    //     }
    //     printf("  min out-deg   : %d\n", min_deg);
    //     printf("  max out-deg   : %d\n", max_deg);
    //     printf("  avg out-deg   : %.3f\n", (double)csr.E / csr.V);
    // }
    timer_start(&t_bellman);
    printf("BellMan_Ford distances : \n");
    float* dist = bellman_GPU_tpe(&csr, 0);
    timer_stop(&t_bellman);
    
    // for(int i=0;i<e.V;i++){
    //     printf("%f  ",dist[i]);
    // }

    double read_ms   = timer_elapsed_ms(&t_edge);
    double csr_ms     = timer_elapsed_ms(&t_csr);
    double bellman_ms = timer_elapsed_ms(&t_bellman);

    printf("\n--- Timing ---\n");
    printf("Read graph   : %.4f ms\n", read_ms);
    printf("Build CSR    : %.4f ms\n", csr_ms);
    printf("BellMan FOrd    : %.4f ms\n", bellman_ms);
    printf("Total (excl. I/O print) : %.4f ms\n", csr_ms + bellman_ms);

    if(dump){
        printf("SSSP Distances : \n");
        for(int i=0;i<csr.V;i++){
            printf("Vertex %d : Distance from source = %.3f\n", i, dist[i]);
        }
    }

    free(e.edges);
    free_csr(&csr);
    free(dist);   

    return 0;
}