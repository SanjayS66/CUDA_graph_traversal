#ifndef BFS_H
#define BFS_H

#include "csr.h"

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

typedef struct {
    int *traversal_order;
    int num_visited;
    bool found;
} BFSResult;

BFSResult bfs(const CSRGraph *csr, int source, int target);
BFSResult bfs_gpu_tpv(const CSRGraph *csr, int source, int target);

#ifdef __cplusplus
}
#endif

#endif /* BFS_H */