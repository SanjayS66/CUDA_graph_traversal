#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "bfs.h"
#include "cuda_utils.h"
#include "csr.h"
#include "timer.h"

__global__ void bfs_expand_tpv_kernel(
    const int *row_offset,
    const int *col_index,
    const int *current_frontier,
    int current_size,
    int *next_frontier,
    int *next_size,
    int *visited) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= current_size) {
        return;
    }

    int u = current_frontier[tid];
    for (int edge = row_offset[u]; edge < row_offset[u + 1]; ++edge) {
        int v = col_index[edge];
        if (atomicCAS(&visited[v], 0, 1) == 0) {
            int position = atomicAdd(next_size, 1);
            next_frontier[position] = v;
        }
    }
}

__global__ void bfs_expand_wpv_kernel(
    const int *row_offset,
    const int *col_index,
    const int *current_frontier,
    int current_size,
    int *next_frontier,
    int *next_size,
    int *visited) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;
    if (warp_id >= current_size) {
        return;
    }

    int u = current_frontier[warp_id];
    for (int edge = row_offset[u] + lane_id;
         edge < row_offset[u + 1];
         edge += 32) {
        int v = col_index[edge];
        if (atomicCAS(&visited[v], 0, 1) == 0) {
            int position = atomicAdd(next_size, 1);
            next_frontier[position] = v;
        }
    }
}

static BFSResult bfs_cpu(const CSRGraph *csr, int source, int target) {
    assert(csr != NULL);
    assert(source >= 0 && source < csr->V);
    assert(target >= 0 && target < csr->V);

    BFSResult result;
    result.traversal_order = (int *)malloc(csr->V * sizeof(int));
    bool *visited = (bool *)calloc(csr->V, sizeof(bool));
    int *queue = (int *)malloc(csr->V * sizeof(int));
    assert(result.traversal_order != NULL);
    assert(visited != NULL);
    assert(queue != NULL);

    result.num_visited = 0;
    result.found = false;
    int front = 0;
    int rear = 0;
    queue[rear++] = source;
    visited[source] = true;

    while (front < rear) {
        int u = queue[front++];
        result.traversal_order[result.num_visited++] = u;
        if (u == target) {
            result.found = true;
            break;
        }

        for (int edge = csr->row_offset[u];
             edge < csr->row_offset[u + 1];
             ++edge) {
            int v = csr->col_index[edge];
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

static BFSResult bfs_gpu(
    const CSRGraph *csr,
    int source,
    int target,
    bool warp_per_vertex) {
    assert(csr != NULL);
    assert(source >= 0 && source < csr->V);
    assert(target >= 0 && target < csr->V);

    const int vertex_count = csr->V;
    DeviceCSR device_csr = copy_csr_to_device(csr);
    int *current_frontier = NULL;
    int *next_frontier = NULL;
    int *next_size = NULL;
    int *visited = NULL;

    CUDA_CHECK(cudaMalloc(&current_frontier, vertex_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&next_frontier, vertex_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&next_size, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&visited, vertex_count * sizeof(int)));
    CUDA_CHECK(cudaMemset(visited, 0, vertex_count * sizeof(int)));

    int one = 1;
    CUDA_CHECK(cudaMemcpy(
        current_frontier, &source, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        visited + source, &one, sizeof(int), cudaMemcpyHostToDevice));

    BFSResult result;
    result.traversal_order = (int *)malloc(vertex_count * sizeof(int));
    assert(result.traversal_order != NULL);
    result.num_visited = 1;
    result.found = source == target;
    result.traversal_order[0] = source;

    int *host_frontier = (int *)malloc(vertex_count * sizeof(int));
    assert(host_frontier != NULL);
    host_frontier[0] = source;
    int current_size = 1;
    const int block_size = 256;

    while (current_size > 0 && !result.found) {
        CUDA_CHECK(cudaMemset(next_size, 0, sizeof(int)));
        int block_count = warp_per_vertex
            ? (current_size + (block_size / 32) - 1) / (block_size / 32)
            : (current_size + block_size - 1) / block_size;

        if (warp_per_vertex) {
            bfs_expand_wpv_kernel<<<block_count, block_size>>>(
                device_csr.row_offset,
                device_csr.col_index,
                current_frontier,
                current_size,
                next_frontier,
                next_size,
                visited);
        } else {
            bfs_expand_tpv_kernel<<<block_count, block_size>>>(
                device_csr.row_offset,
                device_csr.col_index,
                current_frontier,
                current_size,
                next_frontier,
                next_size,
                visited);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        int host_next_size = 0;
        CUDA_CHECK(cudaMemcpy(
            &host_next_size, next_size, sizeof(int), cudaMemcpyDeviceToHost));
        if (host_next_size == 0) {
            break;
        }

        CUDA_CHECK(cudaMemcpy(
            host_frontier,
            next_frontier,
            host_next_size * sizeof(int),
            cudaMemcpyDeviceToHost));

        for (int i = 0; i < host_next_size; ++i) {
            result.traversal_order[result.num_visited++] = host_frontier[i];
            if (host_frontier[i] == target) {
                result.found = true;
            }
        }

        int *temporary = current_frontier;
        current_frontier = next_frontier;
        next_frontier = temporary;
        current_size = host_next_size;
    }

    free(host_frontier);
    free_device_csr(&device_csr);
    CUDA_CHECK(cudaFree(current_frontier));
    CUDA_CHECK(cudaFree(next_frontier));
    CUDA_CHECK(cudaFree(next_size));
    CUDA_CHECK(cudaFree(visited));
    return result;
}

static void free_result(BFSResult *result) {
    free(result->traversal_order);
    result->traversal_order = NULL;
}

static bool valid_result(
    const BFSResult *result,
    int vertex_count,
    int source,
    int target) {
    if (result->num_visited <= 0 ||
        result->num_visited > vertex_count ||
        result->traversal_order[0] != source) {
        return false;
    }

    bool *seen = (bool *)calloc(vertex_count, sizeof(bool));
    if (seen == NULL) {
        return false;
    }
    bool valid = true;
    for (int i = 0; i < result->num_visited; ++i) {
        int vertex = result->traversal_order[i];
        if (vertex < 0 || vertex >= vertex_count || seen[vertex]) {
            valid = false;
            break;
        }
        seen[vertex] = true;
    }
    if (valid && result->found && !seen[target]) {
        valid = false;
    }
    if (valid && !result->found && seen[target]) {
        valid = false;
    }
    free(seen);
    return valid;
}

static bool same_visited_vertices(
    const BFSResult *first,
    const BFSResult *second,
    int vertex_count) {
    bool *first_seen = (bool *)calloc(vertex_count, sizeof(bool));
    bool *second_seen = (bool *)calloc(vertex_count, sizeof(bool));
    if (first_seen == NULL || second_seen == NULL) {
        free(first_seen);
        free(second_seen);
        return false;
    }

    for (int i = 0; i < first->num_visited; ++i) {
        first_seen[first->traversal_order[i]] = true;
    }
    for (int i = 0; i < second->num_visited; ++i) {
        second_seen[second->traversal_order[i]] = true;
    }

    bool same = true;
    for (int i = 0; i < vertex_count; ++i) {
        if (first_seen[i] != second_seen[i]) {
            same = false;
            break;
        }
    }
    free(first_seen);
    free(second_seen);
    return same;
}

static int run_benchmark(
    const CSRGraph *csr,
    int source,
    int target,
    bool dump) {
    Timer timer;
    timer_start(&timer);
    BFSResult cpu = bfs_cpu(csr, source, target);
    timer_stop(&timer);
    double cpu_ms = timer_elapsed_ms(&timer);

    timer_start(&timer);
    BFSResult tpv = bfs_gpu(csr, source, target, false);
    timer_stop(&timer);
    double tpv_ms = timer_elapsed_ms(&timer);

    timer_start(&timer);
    BFSResult wpv = bfs_gpu(csr, source, target, true);
    timer_stop(&timer);
    double wpv_ms = timer_elapsed_ms(&timer);

    bool cpu_valid = valid_result(&cpu, csr->V, source, target);
    bool tpv_valid = valid_result(&tpv, csr->V, source, target);
    bool wpv_valid = valid_result(&wpv, csr->V, source, target);
    bool tpv_matches = cpu.found == tpv.found;
    bool wpv_matches = cpu.found == wpv.found;
    if (!cpu.found) {
        tpv_matches = tpv_matches &&
                      same_visited_vertices(&cpu, &tpv, csr->V);
        wpv_matches = wpv_matches &&
                      same_visited_vertices(&cpu, &wpv, csr->V);
    }

    printf("\n--- BFS Timing ---\n");
    printf("CPU BFS       : %.4f ms\n", cpu_ms);
    printf("GPU TPV BFS   : %.4f ms\n", tpv_ms);
    printf("GPU WPV BFS   : %.4f ms\n", wpv_ms);
    printf("Target Found  : %s\n", cpu.found ? "Yes" : "No");
    printf("TPV Correct   : %s\n", cpu_valid && tpv_valid && tpv_matches ? "Yes" : "No");
    printf("WPV Correct   : %s\n", cpu_valid && wpv_valid && wpv_matches ? "Yes" : "No");

    if (dump) {
        printf("CPU traversal: ");
        for (int i = 0; i < cpu.num_visited; ++i) {
            printf("%d ", cpu.traversal_order[i]);
        }
        printf("\nTPV traversal: ");
        for (int i = 0; i < tpv.num_visited; ++i) {
            printf("%d ", tpv.traversal_order[i]);
        }
        printf("\nWPV traversal: ");
        for (int i = 0; i < wpv.num_visited; ++i) {
            printf("%d ", wpv.traversal_order[i]);
        }
        printf("\n");
    }

    free_result(&cpu);
    free_result(&tpv);
    free_result(&wpv);
    return cpu_valid && tpv_valid && wpv_valid && tpv_matches && wpv_matches ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr,
                "Usage: %s <input_file> <source> <target> "
                "[--undirected] [--dump]\n",
                argv[0]);
        return 1;
    }

    const char *input_path = argv[1];
    int source = atoi(argv[2]);
    int target = atoi(argv[3]);
    int directed = 1;
    bool dump = false;

    for (int i = 4; i < argc; ++i) {
        if (strcmp(argv[i], "--undirected") == 0) {
            directed = 0;
        } else if (strcmp(argv[i], "--dump") == 0) {
            dump = true;
        } else {
            fprintf(stderr, "Unrecognized argument: %s\n", argv[i]);
            return 1;
        }
    }

    EdgeListGraph edge_list = read_graph_from_file(input_path);
    CSRGraph csr = build_csr(&edge_list, directed);
    if (!validate_csr(&csr)) {
        fprintf(stderr, "CSR validation FAILED\n");
        free(edge_list.edges);
        free_csr(&csr);
        return 1;
    }
    if (source < 0 || source >= csr.V || target < 0 || target >= csr.V) {
        fprintf(stderr, "Source and target must be in [0, %d)\n", csr.V);
        free(edge_list.edges);
        free_csr(&csr);
        return 1;
    }

    printf("Graph: %s (V=%d, E=%d, %s)\n",
           input_path,
           csr.V,
           csr.E,
           directed ? "directed" : "undirected");
    printf("Source: %d, Target: %d\n", source, target);
    int status = run_benchmark(&csr, source, target, dump);

    free(edge_list.edges);
    free_csr(&csr);
    return status;
}
