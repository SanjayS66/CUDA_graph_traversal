#include <cstdio>
#include <cstdlib> 
#include <cstring> 
#include "../include/cuda_utils.h"
#include "../include/csr.h"
#include "../include/timer.h"
#include "../include/sssp.h"
#include <cmath>

Timer t_bellman_c,t_edge,t_csr,t_bellman_g;
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

__global__ void relax_kernel(int* row_offset,int* col_idx,float *weights,float* dist,int V,int* changed_flag){
    int u = blockIdx.x * blockDim.x+ threadIdx.x;
    if(u >= V){
        return;
    } 
    for(int k=row_offset[u];k<row_offset[u+1];k++){
        int v = col_idx[k];
        float w = weights[k];
        if(dist[u] + w  < dist[v]){
            atomicMinFloat(&dist[v],dist[u]+w);
            atomicExch(changed_flag,1);
        }
    }
}

__host__ float* bellman_GPU(CSRGraph* csr, int src){
    DeviceCSR d_csr = copy_csr_to_device(csr);

    int V = d_csr.V;
    // int E = d_csr.E;

    float* d_dist;
    size_t size = V*sizeof(float);
    CUDA_CHECK(cudaMalloc((void**)&d_dist, size));
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
    int numBlocks = (V+ blockSize -1)/blockSize;

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

float* bellman_ford(CSRGraph* csr,int src){
    int V = csr->V;
    int E = csr->E;
    int* row_offset = csr->row_offset;
    int* col_idx = csr->col_index;
    float* weights = csr->weights;
    bool changed;

    float* dist = (float*)malloc(V*sizeof(float)); 
    for(int i=0;i<V;i++){
        dist[i] = 1e9;
    }
    dist[src] = 0;

    int i =0 ;
    //main fnc logic
    changed = true;
    while(i < V - 1 && changed){
        changed = false;

        for(int u = 0; u < V; u++){                          // for every vertex u
            for(int k = row_offset[u]; k < row_offset[u+1]; k++){  // for every edge (u -> v)
                int v = col_idx[k];
                float w = weights[k];

                if(dist[u] + w < dist[v]){
                    dist[v] = dist[u] + w;
                    changed = true;
                }
            }
        }

        i++;
    }
    return dist;
}

bool floats_close(float a, float b, float epsilon = 1e-3f) {
    return fabs(a - b) < epsilon;
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
    float* weights = csr.weights;
    int* col_idx = csr.col_index;
    int* row_offsets = csr.row_offset;
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
    printf("CPU implementation");
    timer_start(&t_bellman_c);
    printf("BellMan_Ford distances : \n");
    float* dist_c = bellman_ford(&csr,0);
    timer_stop(&t_bellman_c);
    
    printf("GPU implementation");
    timer_start(&t_bellman_g);
    float* dist_g = bellman_GPU(&csr,0);
    timer_stop(&t_bellman_g);


    double read_ms   = timer_elapsed_ms(&t_edge);
    double csr_ms     = timer_elapsed_ms(&t_csr);
    double bellman_ms_c = timer_elapsed_ms(&t_bellman_c);
    double bellman_ms_g = timer_elapsed_ms(&t_bellman_g);


    printf("\n--- Timing ---\n");
    printf("Read graph   : %.4f ms\n", read_ms);
    printf("Build CSR    : %.4f ms\n", csr_ms);
    printf("CPU BellMan FOrd    : %.4f ms\n", bellman_ms_c);
    printf("GPU BellMan FOrd    : %.4f ms\n", bellman_ms_g);

int mismatches = 0;
    int max_report = 10;   // don't spam the terminal if many mismatches

    for (int i = 0; i < csr.V; i++) {
        if (!floats_close(dist_c[i], dist_g[i])) {
            mismatches++;
            if (mismatches <= max_report) {
                printf("Mismatch at vertex %d: CPU=%.6f GPU=%.6f (diff=%.6f)\n",
                    i, dist_c[i], dist_g[i], fabs(dist_c[i] - dist_g[i]));
            }
        }
    }

    if (mismatches == 0) {
        printf("CORRECTNESS CHECK PASSED: all %d distances match within tolerance\n", csr.V);
    } else {
        printf("CORRECTNESS CHECK FAILED: %d / %d vertices mismatched\n", mismatches, csr.V);
    }
    free(e.edges);
    free_csr(&csr);
    free(dist_c);
    free(dist_g);

    return 0;
}
