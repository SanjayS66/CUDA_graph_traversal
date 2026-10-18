//bfs_cpu.h

#include <cstdio>  
#include <cstdlib> 
#include <cstring> 
#include "csr.h"
#include "bfs.h"
#include "timer.h"


Timer t_bfs,t_edge,t_csr;

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

//Breadth First Search function CPU
BFSResult bfs(CSRGraph* csr, int source,int target) {
    int V = csr->V;
    int* row_offset = csr->row_offset;
    int* col_idx = csr->col_index;
    
    //initialise the result structure arrays and other vars
    BFSResult result;
    result.traversal_order = (int*)malloc(V * sizeof(int));
    result.num_visited = 0;
    result.found = false;

    bool* visited = (bool*)calloc(V, sizeof(bool));
    int* queue = (int*)malloc(V * sizeof(int));
    
    if (!result.traversal_order || !visited || !queue) {
        fprintf(stderr, "Memory allocation failed for BFS\n");
        exit(1);
    }

    int front = 0, rear = 0;
    queue[rear++] = source;
    visited[source] = true;

    while (front < rear) {
        int u = queue[front++];
        
        result.traversal_order[result.num_visited++] = u;
        
        if (u == target) {
            result.found = true;
            break; 
        }
        
        int edge_start = csr->row_offset[u];
        int edge_end = csr->row_offset[u + 1];
        
        for (int i = edge_start; i < edge_end; i++) {
            int v = csr->col_index[i];
            
            if (!visited[v]) {
                visited[v] = true;
                queue[rear++] = v;
            }
        }
    }
    free(visited);
    free(queue);
    
    return result;


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
    timer_start(&t_bfs);
    BFSResult res = bfs(&csr, 0, csr.V - 1); // Example: BFS from vertex 0 to the last vertex
    timer_stop(&t_bfs);
    
    // for(int i=0;i<e.V;i++){
    //     printf("%f  ",dist[i]);
    // }

    double read_ms   = timer_elapsed_ms(&t_edge);
    double csr_ms     = timer_elapsed_ms(&t_csr);
    double bfs_ms = timer_elapsed_ms(&t_bfs);

    printf("\n--- Timing ---\n");
    printf("Read graph   : %.4f ms\n", read_ms);
    printf("Build CSR    : %.4f ms\n", csr_ms);
    printf("BFS          : %.4f ms\n", bfs_ms);
    printf("Target Found : %s\n", res.found ? "Yes" : "No");
    printf("Total (excl. I/O print) : %.4f ms\n", csr_ms + bfs_ms);

    for(int i=0;i<csr.V;i++){
        printf("%d  ",res.traversal_order[i]);
    }
    free(e.edges);
    free_csr(&csr);

    return 0;
}

