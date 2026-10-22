// Thread-per-frontier-vertex, level-synchronous BFS using two device queues.
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "bfs.h"
#include "cuda_utils.h"
#include "timer.h"

__global__ void bfs_expand_kernel(
    const int *row_offset,
    const int *col_index,
    const int *current_frontier,
    int current_size,
    int *next_frontier,
    int *next_size,
    int *visited,
    int target,
    int *found
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= current_size) {
        return;
    }

    int u = current_frontier[tid];
    for (int edge = row_offset[u]; edge < row_offset[u + 1]; edge++) {
        int v = col_index[edge];
        if (atomicCAS(&visited[v], 0, 1) == 0) {
            int position = atomicAdd(next_size, 1);
            next_frontier[position] = v;
            if (v == target) {
                atomicExch(found, 1);
            }
        }
    }
}

BFSResult bfs_gpu_tpv(const CSRGraph *csr, int source, int target) {
    if (csr == nullptr || csr->V <= 0 ||
        source < 0 || source >= csr->V ||
        target < 0 || target >= csr->V) {
        fprintf(stderr, "Invalid BFS source or target\n");
        exit(EXIT_FAILURE);
    }

    const int V = csr->V;
    DeviceCSR d_csr = copy_csr_to_device(csr);
    int *d_current_frontier = nullptr;
    int *d_next_frontier = nullptr;
    int *d_next_size = nullptr;
    int *d_visited = nullptr;
    int *d_found = nullptr;

    CUDA_CHECK(cudaMalloc(&d_current_frontier, V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_frontier, V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_size, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_visited, V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_found, sizeof(int)));

    CUDA_CHECK(cudaMemset(d_visited, 0, V * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_found, 0, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(
        d_current_frontier, &source, sizeof(int), cudaMemcpyHostToDevice));
    int one = 1;
    CUDA_CHECK(cudaMemcpy(
        d_visited + source, &one, sizeof(int), cudaMemcpyHostToDevice));

    BFSResult result;
    result.traversal_order = static_cast<int *>(malloc(V * sizeof(int)));
    result.num_visited = 0;
    result.found = (source == target);
    if (result.traversal_order == nullptr) {
        fprintf(stderr, "Memory allocation failed for BFS result\n");
        exit(EXIT_FAILURE);
    }
    result.traversal_order[result.num_visited++] = source;

    int *host_frontier = static_cast<int *>(malloc(V * sizeof(int)));
    if (host_frontier == nullptr) {
        fprintf(stderr, "Memory allocation failed for BFS frontier\n");
        free(result.traversal_order);
        exit(EXIT_FAILURE);
    }

    int current_size = 1;
    int next_size = 0;
    const int block_size = 256;

    while (current_size > 0 && !result.found) {
        CUDA_CHECK(cudaMemset(d_next_size, 0, sizeof(int)));

        int block_count = (current_size + block_size - 1) / block_size;
        bfs_expand_kernel<<<block_count, block_size>>>(
            d_csr.row_offset,
            d_csr.col_index,
            d_current_frontier,
            current_size,
            d_next_frontier,
            d_next_size,
            d_visited,
            target,
            d_found);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        int h_found = 0;
        CUDA_CHECK(cudaMemcpy(
            &next_size, d_next_size, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            &h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost));
        result.found = h_found != 0;

        if (next_size > 0) {
            CUDA_CHECK(cudaMemcpy(
                host_frontier,
                d_next_frontier,
                next_size * sizeof(int),
                cudaMemcpyDeviceToHost));

            int copy_count = next_size;
            if (copy_count > V - result.num_visited) {
                copy_count = V - result.num_visited;
            }
            memcpy(
                result.traversal_order + result.num_visited,
                host_frontier,
                copy_count * sizeof(int));
            result.num_visited += copy_count;
        }

        if (result.found || next_size == 0) {
            break;
        }

        int *temporary = d_current_frontier;
        d_current_frontier = d_next_frontier;
        d_next_frontier = temporary;
        current_size = next_size;
    }

    free(host_frontier);
    free_device_csr(&d_csr);
    CUDA_CHECK(cudaFree(d_current_frontier));
    CUDA_CHECK(cudaFree(d_next_frontier));
    CUDA_CHECK(cudaFree(d_next_size));
    CUDA_CHECK(cudaFree(d_visited));
    CUDA_CHECK(cudaFree(d_found));
    return result;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input_file> [--undirected] [--dump]\n", argv[0]);
        return 1;
    }

    char *input_path = argv[1];
    int directed = 1;
    int dump = 0;
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

    Timer t_bfs, t_edge, t_csr;
    timer_start(&t_edge);
    EdgeListGraph edge_list = read_graph_from_file(input_path);
    timer_stop(&t_edge);
    printf("Read graph: V=%d E=%d from %s\n",
           edge_list.V, edge_list.E, input_path);

    timer_start(&t_csr);
    CSRGraph csr = build_csr(&edge_list, directed);
    if (!validate_csr(&csr)) {
        fprintf(stderr, "CSR validation FAILED\n");
        free(edge_list.edges);
        free_csr(&csr);
        return 1;
    }
    timer_stop(&t_csr);
    printf("CSR built and validated OK\n");

    timer_start(&t_bfs);
    BFSResult result = bfs_gpu_tpv(&csr, 0, csr.V - 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    timer_stop(&t_bfs);

    printf("\n--- Timing ---\n");
    printf("Read graph   : %.4f ms\n", timer_elapsed_ms(&t_edge));
    printf("Build CSR    : %.4f ms\n", timer_elapsed_ms(&t_csr));
    printf("BFS GPU TPV  : %.4f ms\n", timer_elapsed_ms(&t_bfs));
    printf("Target Found : %s\n", result.found ? "Yes" : "No");
    if (dump) {
        printf("BFS traversal: ");
        for (int i = 0; i < result.num_visited; i++) {
            printf("%d ", result.traversal_order[i]);
        }
        printf("\n");
    }

    free(result.traversal_order);
    free(edge_list.edges);
    free_csr(&csr);
    return 0;
}
