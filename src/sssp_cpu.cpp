#include <cstdio>  
#include <cstdlib> 
#include <cstring> 
#include "csr.h"

#include <queue>
#include <vector>
#include <utility> 

#include <chrono>
using Clock = std::chrono::steady_clock;

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

//DJikstras function 
float* djikstras(CSRGraph* csr,int src){
    int V = csr->V;
    int E = csr->E;
    int* row_offset = csr->row_offset;
    int* col_idx = csr->col_index;
    float* weights = csr->weights;
    
    //priority queue
    using NodeDistPair = std::pair<float, int>;

    std::priority_queue<
        NodeDistPair,                //tells what data is stored 
        std::vector<NodeDistPair>,   //tells to store elements of pq as cpp vector
        std::greater<NodeDistPair>   //flips default max heap to use min heap
    > pq;
    pq.push({0.0f, src});

    float* dist = (float*)malloc(V*sizeof(float)); 
    for(int i=0;i<V;i++){
        dist[i] = 1e9;
    }
    dist[src] = 0;


    //main fnc logic
    while(!pq.empty()){
        float distance = pq.top().first;
        int u = pq.top().second;
        pq.pop();

        if (distance > dist[u]) continue;

        for(int i = row_offset[u]; i < row_offset[u+1]; i++){
            int v = col_idx[i];
            if(dist[v] > distance + weights[i]){
                dist[v] = distance + weights[i];
                pq.push({dist[u],v});
            }
        }
    }

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

    auto t0 = Clock::now();
    EdgeListGraph e = return_edgelist(input_path);
    auto t1 = Clock::now();
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
    CSRGraph csr = return_csr(&e,directed);
    auto t2 = Clock::now();
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
    printf("Djikstras distances : \n");
    float* dist = djikstras(&csr,0);
    auto t3 = Clock::now();
    // for(int i=0;i<e.V;i++){
    //     printf("%f  ",dist[i]);
    // }

    double read_ms   = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double csr_ms     = std::chrono::duration<double, std::milli>(t2 - t1).count();
    double dijkstra_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    printf("\n--- Timing ---\n");
    printf("Read graph   : %.4f ms\n", read_ms);
    printf("Build CSR    : %.4f ms\n", csr_ms);
    printf("Dijkstra     : %.4f ms\n", dijkstra_ms);
    printf("Total (excl. I/O print) : %.4f ms\n", csr_ms + dijkstra_ms);

    free(e.edges);
    free_csr(&csr);

    return 0;
}

