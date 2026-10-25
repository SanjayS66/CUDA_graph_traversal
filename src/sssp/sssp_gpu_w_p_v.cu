#include <cstdio>
#include <cstdlib> 
#include <cstring> 
#include "cuda_utils.h"
#include "csr.h"
#include "timer.h"
#include "sssp.h"

__global__ void relax_kernel(int* row_offset, int* col_idx, float *weights, float* dist, int V, int* changed_flag){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int u = tid / 32;        // Global warp ID = vertex u
    int lane_id = tid % 32;  // Lane index inside the warp (0..31)

    if (u >= V) {
        return;
    }

    // Cooperative stride: warp threads stride in steps of 32
    for (int k = row_offset[u] + lane_id; k < row_offset[u + 1]; k += 32) {
        int v = col_idx[k];
        float w = weights[k];
        if (dist[u] + w < dist[v]) {
            atomicMinFloat(&dist[v], dist[u] + w);
            atomicExch(changed_flag, 1);
        }
    }
}


__host__ float* bellman_GPU_wpv(const CSRGraph* csr, int src){
    DeviceCSR d_csr = copy_csr_to_device(csr);

    int V = d_csr.V;
    // int E = d_csr.E;

    float* d_dist;
    size_t size = V*sizeof(float);
    CUDA_CHECK(cudaMalloc((void**)&d_dist,size));

    int* d_changed;
    CUDA_CHECK(cudaMalloc((void**)&d_changed, sizeof(int)));

    //initialise hte d_dist array into 1e-9 vale
    float* h_init = (float*)malloc(V * sizeof(float));
    for(int i=0;i<V;i++){
        h_init[i] = 1e9;
    }
    h_init[src] = 0;
    CUDA_CHECK(cudaMemcpy(d_dist,h_init,V*sizeof(float),cudaMemcpyHostToDevice));
    free(h_init);

    int blockSize = 256;
    int warpsPerBlock = blockSize / 32;
    int numBlocks = (V + warpsPerBlock - 1) / warpsPerBlock;

    int i=0;
    int h_changed = 1;
    while(i<V-1 && h_changed){
        h_changed = 0;
        CUDA_CHECK(cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice));
        relax_kernel<<<numBlocks, blockSize>>>(d_csr.row_offset, d_csr.col_index,
                                                 d_csr.weights, d_dist, V, d_changed);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        i++;
    }
    printf("Bellman-Ford GPU converged in %d rounds\n", i);
    float* dist = (float*)malloc(V*sizeof(float));
    CUDA_CHECK(cudaMemcpy(dist,d_dist,V * sizeof(float),cudaMemcpyDeviceToHost));
    
    free_device_csr(&d_csr);
    cudaFree(d_dist);
    cudaFree(d_changed);
    return dist;
}


//main fnc
int main(int argc, char **argv) {
    Timer t_bellman,t_edge,t_csr;

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
    EdgeListGraph e = read_graph_from_file(input_path);
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
    CSRGraph csr = build_csr(&e, directed);
    if (!validate_csr(&csr)) {
        fprintf(stderr, "CSR validation FAILED\n");
        free(e.edges);
        free_csr(&csr);
        return 1;
    }
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
    float* dist = bellman_GPU_wpv(&csr,0);
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