# CUDA Graph Traversal

This project implements graph preprocessing and graph traversal algorithms for
CPU and CUDA GPU execution. Graphs are loaded from a weighted edge-list file,
converted to **Compressed Sparse Row (CSR)** format, and then processed using:

- Breadth-First Search (BFS)
- Single-Source Shortest Paths (SSSP) using Bellman-Ford
- CPU and GPU implementations for comparison

The GPU implementations use different mappings of graph work to CUDA threads
and warps.

## Features

- Shared graph loading and CSR construction for all algorithms
- Directed and undirected graph support
- CSR validation and optional CSR dumping
- CPU BFS
- GPU BFS:
  - Thread-per-frontier-vertex (TPV)
  - Warp-per-frontier-vertex (WPV)
- CPU Bellman-Ford SSSP
- GPU Bellman-Ford:
  - Thread-per-vertex (TPV)
  - Thread-per-edge (TPE)
  - Warp-per-vertex (WPV)
- Unified BFS and SSSP benchmark programs
- Timing of graph loading, CSR construction, and algorithm execution

## Repository layout

```text
PCAP_implementation/
├── commands.txt
├── README.md
├── PCAP_Synopsis_Draft.pdf
│
├── include/
│   ├── bfs.h
│   ├── csr.h
│   ├── cuda_utils.h
│   ├── sssp.h
│   └── timer.h
│
├── src/
│   ├── bfs/
│   │   ├── bfs_cpu.cpp
│   │   ├── bfs_gpu_t_p_v.cu
│   │   └── bfs_gpu_w_p_v.cu
│   ├── common_stuff/
│   │   ├── csr.cpp
│   │   ├── cuda_utils.cu
│   │   └── timer.cpp
│   └── sssp/
│       ├── sssp_cpu.cpp
│       ├── sssp_gpu_t_p_e.cu
│       ├── sssp_gpu_t_p_v.cu
│       └── sssp_gpu_w_p_v.cu
│
├── test/
│   ├── test_bfs.cu
│   ├── test_csr.cpp
│   └── test_sssp.cu
│
└── graphs/
    ├── sample_graph.txt
    ├── graph1.edgelist
    └── graph_gen.py
```

The `build/` directory contains generated executables and object files. It is
not part of the source tree.

## Input graph format

Each input file is a weighted edge list with the following format:

```text
<number_of_vertices> <number_of_edges>
<source_vertex> <destination_vertex> <weight>
<source_vertex> <destination_vertex> <weight>
...
```

For example:

```text
5 6
0 1 1
1 2 2
0 3 4
2 3 1
3 4 3
1 4 10
```

Vertex IDs must be in the range `[0, V)`. The graph loader validates the
header, edge count, edge endpoints, and edge values while reading the file.

Input edges are directed by default. With the `--undirected` option, every
input edge `(u, v, w)` is represented in CSR as both `(u, v, w)` and
`(v, u, w)`.

## CSR representation

CSR stores the graph using three arrays:

| Array | Size | Purpose |
|---|---:|---|
| `row_offset` | `V + 1` | Start and end offsets of each vertex's adjacency range |
| `col_index` | `E` | Destination vertex for each stored edge |
| `weights` | `E` | Weight corresponding to each entry in `col_index` |

The outgoing neighbors of vertex `u` are stored in:

```text
row_offset[u] ... row_offset[u + 1] - 1
```

CSR construction is centralized in
[`src/common_stuff/csr.cpp`](src/common_stuff/csr.cpp), and its public
declarations are in [`include/csr.h`](include/csr.h). CPU and GPU algorithms
all use this shared implementation.

CUDA device allocation and host-to-device CSR copies are handled separately by
[`src/common_stuff/cuda_utils.cu`](src/common_stuff/cuda_utils.cu).

## Algorithms

### BFS

BFS starts from a source vertex and explores the graph level by level until the
target is found or all reachable vertices have been visited.

The GPU implementations use a frontier-based traversal:

- **TPV:** one CUDA thread processes one frontier vertex.
- **WPV:** one CUDA warp processes one frontier vertex, with lanes cooperating
  over that vertex's outgoing edges.

GPU traversal order may differ between runs because frontier insertion uses
atomic operations. Correctness should therefore be evaluated using reachability
and the visited set rather than requiring an identical traversal order.

### SSSP

The SSSP implementations use the Bellman-Ford relaxation algorithm to compute
distances from one source vertex:

- **CPU:** sequential edge relaxation
- **GPU TPV:** work mapped to vertices
- **GPU TPE:** work mapped to edges
- **GPU WPV:** work mapped to warps and vertices

The SSSP input is weighted. The source vertex defaults to `0` for the
standalone implementations. The unified benchmark accepts a source through
`--src`.

## Requirements

- Linux or another Unix-like environment
- A C++11-compatible compiler such as `g++`
- NVIDIA CUDA Toolkit with `nvcc`
- A CUDA-capable NVIDIA GPU for GPU programs

The CPU programs and CSR test can run without a CUDA-capable GPU. GPU programs
require a working CUDA driver and compatible device.

## Build and run

The complete build and execution commands are maintained in
[`commands.txt`](commands.txt). Run commands from the repository root:

```bash
cd /path/to/PCAP_implementation
mkdir -p build
```

### CSR test

```bash
g++ -std=c++11 -Wall -Wextra -Iinclude \
  test/test_csr.cpp \
  src/common_stuff/csr.cpp \
  -o build/test_csr

./build/test_csr graphs/sample_graph.txt
```

### CPU BFS

```bash
g++ -std=c++11 -Wall -Wextra -Iinclude \
  src/bfs/bfs_cpu.cpp \
  src/common_stuff/csr.cpp \
  src/common_stuff/timer.cpp \
  -o build/bfs_cpu

./build/bfs_cpu graphs/sample_graph.txt
```

### GPU BFS

Thread-per-frontier-vertex:

```bash
nvcc -std=c++11 -Iinclude \
  src/bfs/bfs_gpu_t_p_v.cu \
  src/common_stuff/csr.cpp \
  src/common_stuff/cuda_utils.cu \
  src/common_stuff/timer.cpp \
  -o build/bfs_gpu_tpv

./build/bfs_gpu_tpv graphs/sample_graph.txt
```

Warp-per-frontier-vertex:

```bash
nvcc -std=c++11 -Iinclude \
  src/bfs/bfs_gpu_w_p_v.cu \
  src/common_stuff/csr.cpp \
  src/common_stuff/cuda_utils.cu \
  src/common_stuff/timer.cpp \
  -o build/bfs_gpu_wpv

./build/bfs_gpu_wpv graphs/sample_graph.txt
```

### CPU SSSP

```bash
g++ -std=c++11 -Wall -Wextra -Iinclude \
  src/sssp/sssp_cpu.cpp \
  src/common_stuff/csr.cpp \
  src/common_stuff/timer.cpp \
  -o build/sssp_cpu

./build/sssp_cpu graphs/sample_graph.txt
```

### GPU SSSP

Thread-per-vertex:

```bash
nvcc -std=c++11 -Iinclude \
  src/sssp/sssp_gpu_t_p_v.cu \
  src/common_stuff/csr.cpp \
  src/common_stuff/cuda_utils.cu \
  src/common_stuff/timer.cpp \
  -o build/sssp_gpu_tpv

./build/sssp_gpu_tpv graphs/sample_graph.txt
```

Thread-per-edge:

```bash
nvcc -std=c++11 -Iinclude \
  src/sssp/sssp_gpu_t_p_e.cu \
  src/common_stuff/csr.cpp \
  src/common_stuff/cuda_utils.cu \
  src/common_stuff/timer.cpp \
  -o build/sssp_gpu_tpe

./build/sssp_gpu_tpe graphs/sample_graph.txt
```

Warp-per-vertex:

```bash
nvcc -std=c++11 -Iinclude \
  src/sssp/sssp_gpu_w_p_v.cu \
  src/common_stuff/csr.cpp \
  src/common_stuff/cuda_utils.cu \
  src/common_stuff/timer.cpp \
  -o build/sssp_gpu_wpv

./build/sssp_gpu_wpv graphs/sample_graph.txt
```

## Unified benchmarks

### BFS benchmark

The unified BFS benchmark runs CPU BFS, GPU TPV BFS, and GPU WPV BFS for one
graph, source, and target:

```bash
nvcc -std=c++11 -Xcompiler=-Wall,-Wextra -Iinclude \
  test/test_bfs.cu \
  src/common_stuff/csr.cpp \
  src/common_stuff/cuda_utils.cu \
  src/common_stuff/timer.cpp \
  -o build/test_bfs

./build/test_bfs graphs/sample_graph.txt 0 4
./build/test_bfs graphs/sample_graph.txt 0 4 --undirected --dump
```

Usage:

```text
./build/test_bfs <input_file> <source> <target> [--undirected] [--dump]
```

The benchmark reports execution times and checks that the CPU and GPU results
agree on whether the target is reachable and on the visited vertices.

### SSSP benchmark

The unified SSSP benchmark runs the CPU implementation and all three GPU
implementations:

```bash
nvcc -std=c++11 -Iinclude \
  test/test_sssp.cu \
  src/common_stuff/csr.cpp \
  src/common_stuff/cuda_utils.cu \
  src/common_stuff/timer.cpp \
  -o build/benchmark_sssp

./build/benchmark_sssp graphs/graph1.edgelist
./build/benchmark_sssp graphs/graph1.edgelist --undirected --dump --src 0
```

Usage:

```text
./build/benchmark_sssp <input_file>
    [--undirected] [--dump] [--src <vertex>]
```

The benchmark prints timing information and the computed distances. Use
`--dump` when comparing individual vertex distances on a small graph.

## Available graph files

The repository includes small and larger graph inputs under `graphs/`.

| File | Purpose |
|---|---|
| `sample_graph.txt` | Small graph for quick correctness checks |
| `graph1.edgelist` | Larger graph for benchmark runs |
| `graph_gen.py` | Utility for generating graph files |
| `graph_large_sparse.edgelist` | Large sparse validation graph |
| `graph_largest_sparse.edgelist` | Very large sparse validation graph |
| `gplus_combined.edgelist` | Real-world graph input |
| `roadNet-CA.edgelist` | Real-world road-network graph input |

Large graph runs require substantially more host memory and GPU memory than the
sample graph. Begin with the sample graph, then move to larger inputs after
the correctness checks pass.


### Comparing GPU BFS output

The order in which vertices appear in a GPU BFS traversal can differ from the
CPU order. Compare reachability and visited vertices rather than requiring the
same order.

## Validation results

The following results were obtained using the unified BFS and SSSP benchmarks
on the available NVIDIA GPU. Timings are included as reference measurements;
they can vary with GPU load, driver version, compiler version, and system
configuration.

### BFS on real-world graphs

Both GPU implementations matched the CPU correctness checks and found the
requested target.

#### `gplus_combined.edgelist`

Command:

```bash
./build/test_bfs graphs/gplus_combined.edgelist 0 102099
```

Graph: `V = 102100`, `E = 12113501`, directed.

| Implementation | Time |
|---|---:|
| CPU BFS | 19.5651 ms |
| GPU TPV BFS | 204.2895 ms |
| GPU WPV BFS | 22.5946 ms |

Result: target found, TPV correct, and WPV correct.

#### `roadNet-CA.edgelist`

Command:

```bash
./build/test_bfs graphs/roadNet-CA.edgelist 0 1971280
```

Graph: `V = 1971281`, `E = 5533214`, directed.

| Implementation | Time |
|---|---:|
| CPU BFS | 14.6368 ms |
| GPU TPV BFS | 188.3604 ms |
| GPU WPV BFS | 17.2368 ms |

Result: target found, TPV correct, and WPV correct.

### SSSP on real-world graphs

The unified SSSP benchmark verified all GPU implementations against the CPU
Bellman-Ford baseline. Verification passed for 100% of the compared
distances in both runs.

#### `roadNet-CA.edgelist`

Command:

```bash
./build/test_sssp graphs/roadNet-CA.edgelist
```

Graph: `V = 1971281`, `E = 5533214`, directed. Source vertex: `0`.

| Strategy | Rounds | Time | Speedup versus CPU |
|---|---:|---:|---:|
| CPU Bellman-Ford | 264 | 4945.2346 ms | 1.00x |
| GPU TPV | 394 | 258.9295 ms | 19.10x |
| GPU TPE | 527 | 435.6605 ms | 11.35x |
| GPU WPV | 485 | 1305.4254 ms | 3.79x |

Verification: all GPU implementations passed.

#### `gplus_combined.edgelist`

Command:

```bash
./build/test_sssp graphs/gplus_combined.edgelist
```

Graph: `V = 102100`, `E = 12113501`, directed. Source vertex: `0`.

| Strategy | Rounds | Time | Speedup versus CPU |
|---|---:|---:|---:|
| CPU Bellman-Ford | 2 | 45.1010 ms | 1.00x |
| GPU TPV | 5 | 92.2413 ms | 0.49x |
| GPU TPE | 7 | 66.9571 ms | 0.67x |
| GPU WPV | 6 | 27.3488 ms | 1.65x |

Verification: all GPU implementations passed.

