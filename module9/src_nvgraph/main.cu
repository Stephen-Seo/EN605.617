#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <vector>

#include <cuda_device_runtime_api.h>
#include <nvgraph.h>

#include "arg_parse.h"

void SetUpGraph(nvgraphHandle_t *handle, nvgraphGraphDescr_t *graph,
                nvgraphCSCTopology32I_st *CSC_input,
                bool randomizeWeights = false,
                bool seed_random_with_time = false) {
  CSC_input->nvertices = 8;
  CSC_input->nedges = 14;
  CSC_input->destination_offsets =
      (int *)malloc(sizeof(int) * (CSC_input->nvertices + 1));
  CSC_input->destination_offsets[CSC_input->nvertices] = CSC_input->nedges;
  CSC_input->source_indices = (int *)malloc(sizeof(int) * CSC_input->nedges);
  float *edge_weights = (float *)malloc(sizeof(float) * CSC_input->nedges);

  // first vertex array will be result output of shortest paths
  // first edge array will be edge weights
  cudaDataType_t set_type = CUDA_R_32F;

  // set edges
  /*
0 -> 1
0 -> 2
0 -> 3
1 -> 4
1 -> 5
2 -> 4
2 -> 5
2 -> 6
3 -> 5
3 -> 6
4 -> 7
5 -> 7
6 -> 7
7 -> 0

All edges go to the right:
          1  - 4
        /    \/    \
      0 - 2  -  5 - 7
        \    /\    /
          3  - 6
7 has edge back to 0 since CSC by design must have every vertex have an incoming
edge (at least by my understanding of it)
   */

  CSC_input->destination_offsets[0] = 0;
  CSC_input->destination_offsets[1] = 1;
  CSC_input->destination_offsets[2] = 2;
  CSC_input->destination_offsets[3] = 3;
  CSC_input->destination_offsets[4] = 4;
  CSC_input->destination_offsets[5] = 6;
  CSC_input->destination_offsets[6] = 9;
  CSC_input->destination_offsets[7] = 11;
  // index 8 is already set earlier

  CSC_input->source_indices[0] = 7;
  CSC_input->source_indices[1] = 0;
  CSC_input->source_indices[2] = 0;
  CSC_input->source_indices[3] = 0;
  CSC_input->source_indices[4] = 1;
  CSC_input->source_indices[5] = 2;
  CSC_input->source_indices[6] = 1;
  CSC_input->source_indices[7] = 2;
  CSC_input->source_indices[8] = 3;
  CSC_input->source_indices[9] = 2;
  CSC_input->source_indices[10] = 3;
  CSC_input->source_indices[11] = 4;
  CSC_input->source_indices[12] = 5;
  CSC_input->source_indices[13] = 6;

  if (randomizeWeights) {
    if (seed_random_with_time) {
      std::srand(std::time(nullptr));
    } else {
      // seeded to 0 for determinism
      std::srand(0);
    }
    std::cout << "Randomized weights ("
              << (seed_random_with_time ? "seeded with time" : "seeded with 0")
              << "):\n";
    for (unsigned int i = 0; i < CSC_input->nedges; ++i) {
      edge_weights[i] = (float)(rand() % 10 + 1) / 10.0F;
      std::cout << "  edge " << std::setw(2) << i << ": " << std::fixed
                << std::setprecision(1) << edge_weights[i] << '\n';
    }
  } else {
    edge_weights[0] = 1.0F;
    edge_weights[1] = 0.3F;
    edge_weights[2] = 0.2F;
    edge_weights[3] = 0.3F;
    edge_weights[4] = 0.3F;
    edge_weights[5] = 0.4F;
    edge_weights[6] = 0.4F;
    edge_weights[7] = 0.5F;
    edge_weights[8] = 0.4F;
    edge_weights[9] = 0.4F;
    edge_weights[10] = 0.3F;
    edge_weights[11] = 0.4F;
    edge_weights[12] = 0.4F;
    edge_weights[13] = 0.5F;
  }

  nvgraphCreate(handle);

  nvgraphCreate(handle);
  nvgraphCreateGraphDescr(*handle, graph);

  nvgraphSetGraphStructure(*handle, *graph, CSC_input, NVGRAPH_CSC_32);

  // Only one set of Vertex data will hold the result
  nvgraphAllocateVertexData(*handle, *graph, 1, &set_type);

  // Only one set of Edge data will hold the weights
  nvgraphAllocateEdgeData(*handle, *graph, 1, &set_type);
  nvgraphSetEdgeData(*handle, *graph, edge_weights, 0);

  // edge weights have been stored on device, free host data
  free(edge_weights);
}

void SetUpGraphAlt(nvgraphHandle_t *handle, nvgraphGraphDescr_t *graph,
                   nvgraphCSCTopology32I_st *CSC_input,
                   bool randomizeWeights = false,
                   bool seed_random_with_time = false) {
  CSC_input->nvertices = 11;
  CSC_input->nedges = 21;
  CSC_input->destination_offsets =
      (int *)malloc(sizeof(int) * (CSC_input->nvertices + 1));
  CSC_input->destination_offsets[CSC_input->nvertices] = CSC_input->nedges;
  CSC_input->source_indices = (int *)malloc(sizeof(int) * CSC_input->nedges);
  float *edge_weights = (float *)malloc(sizeof(float) * CSC_input->nedges);

  // first vertex array will be result output of shortest paths
  // first edge array will be edge weights
  cudaDataType_t set_type = CUDA_R_32F;

  // set edges
  /*
0 -> 1
0 -> 2
0 -> 3
1 -> 4
1 -> 5
2 -> 4
2 -> 5
2 -> 6
3 -> 5
3 -> 6
4 -> 7
4 -> 8
5 -> 7
5 -> 8
5 -> 9
6 -> 8
6 -> 9
7 -> 10
8 -> 10
9 -> 10
10 -> 0

All edges go to the right:
          1  -  4  -  7  \
        /    \/    \/
      0 - 2  -  5  -  8  -  10
        \    /\    /\
          3  -  6  -  9  /
10 has edge back to 0 since CSC by design must have every vertex have an
incoming edge (at least by my understanding of it)
   */

  CSC_input->destination_offsets[0] = 0;
  CSC_input->destination_offsets[1] = 1;
  CSC_input->destination_offsets[2] = 2;
  CSC_input->destination_offsets[3] = 3;
  CSC_input->destination_offsets[4] = 4;
  CSC_input->destination_offsets[5] = 6;
  CSC_input->destination_offsets[6] = 9;
  CSC_input->destination_offsets[7] = 11;
  CSC_input->destination_offsets[8] = 13;
  CSC_input->destination_offsets[9] = 16;
  CSC_input->destination_offsets[10] = 18;
  // index 11 is already set earlier

  CSC_input->source_indices[0] = 10;
  CSC_input->source_indices[1] = 0;
  CSC_input->source_indices[2] = 0;
  CSC_input->source_indices[3] = 0;
  CSC_input->source_indices[4] = 1;
  CSC_input->source_indices[5] = 2;
  CSC_input->source_indices[6] = 1;
  CSC_input->source_indices[7] = 2;
  CSC_input->source_indices[8] = 3;
  CSC_input->source_indices[9] = 2;
  CSC_input->source_indices[10] = 3;
  CSC_input->source_indices[11] = 4;
  CSC_input->source_indices[12] = 5;
  CSC_input->source_indices[13] = 4;
  CSC_input->source_indices[14] = 5;
  CSC_input->source_indices[15] = 6;
  CSC_input->source_indices[16] = 5;
  CSC_input->source_indices[17] = 6;
  CSC_input->source_indices[18] = 7;
  CSC_input->source_indices[19] = 8;
  CSC_input->source_indices[20] = 9;

  if (randomizeWeights) {
    if (seed_random_with_time) {
      std::srand(std::time(nullptr));
    } else {
      // seeded to 0 for determinism
      std::srand(0);
    }
    std::cout << "Randomized weights ("
              << (seed_random_with_time ? "seeded with time" : "seeded with 0")
              << "):\n";
    for (unsigned int i = 0; i < CSC_input->nedges; ++i) {
      edge_weights[i] = (float)(rand() % 10 + 1) / 10.0F;
      std::cout << "  edge " << std::setw(2) << i << ": " << std::fixed
                << std::setprecision(1) << edge_weights[i] << '\n';
    }
  } else {
    edge_weights[0] = 1.0F;
    edge_weights[1] = 0.3F;
    edge_weights[2] = 0.2F;
    edge_weights[3] = 0.3F;
    edge_weights[4] = 0.3F;
    edge_weights[5] = 0.4F;
    edge_weights[6] = 0.4F;
    edge_weights[7] = 0.5F;
    edge_weights[8] = 0.4F;
    edge_weights[9] = 0.4F;
    edge_weights[10] = 0.3F;
    edge_weights[11] = 0.4F;
    edge_weights[12] = 0.3F;
    edge_weights[13] = 0.4F;
    edge_weights[14] = 0.5F;
    edge_weights[15] = 0.4F;
    edge_weights[16] = 0.4F;
    edge_weights[17] = 0.3F;
    edge_weights[18] = 0.5F;
    edge_weights[19] = 0.5F;
    edge_weights[20] = 0.4F;
  }

  nvgraphCreate(handle);

  nvgraphCreate(handle);
  nvgraphCreateGraphDescr(*handle, graph);

  nvgraphSetGraphStructure(*handle, *graph, CSC_input, NVGRAPH_CSC_32);

  // Only one set of Vertex data will hold the result
  nvgraphAllocateVertexData(*handle, *graph, 1, &set_type);

  // Only one set of Edge data will hold the weights
  nvgraphAllocateEdgeData(*handle, *graph, 1, &set_type);
  nvgraphSetEdgeData(*handle, *graph, edge_weights, 0);

  // edge weights have been stored on device, free host data
  free(edge_weights);
}

void CleanUpGraph(nvgraphHandle_t *handle, nvgraphGraphDescr_t *graph,
                  nvgraphCSCTopology32I_st *CSC_input) {
  nvgraphDestroyGraphDescr(*handle, *graph);
  nvgraphDestroy(*handle);

  free(CSC_input->source_indices);
  free(CSC_input->destination_offsets);
}

void DoPrintOutRun(const Args &args, nvgraphCSCTopology32I_st &CSC_input,
                   nvgraphHandle_t *handle, nvgraphGraphDescr_t *graph) {
  if (args.use_alternate_graph) {
    SetUpGraphAlt(handle, graph, &CSC_input, args.randomize_weights,
                  args.seed_random_with_time);
  } else {
    SetUpGraph(handle, graph, &CSC_input, args.randomize_weights,
               args.seed_random_with_time);
  }

  // Check shortest path starting at 0
  int source_vert = 0;
  nvgraphSssp(*handle, *graph, 0, &source_vert, 0);

  // Get result data from vertex set
  std::vector<float> output_data(CSC_input.nvertices);
  nvgraphGetVertexData(*handle, *graph, output_data.data(), 0);

  std::cout << "Got result values:\n";
  for (unsigned int i = 0; i < CSC_input.nvertices; ++i) {
    std::cout << std::setw(2) << i << ": " << std::setw(5) << std::fixed
              << std::setprecision(2) << output_data[i] << std::endl;
  }
}

void DoTimedRuns(const Args &args, nvgraphCSCTopology32I_st &CSC_input,
                 nvgraphHandle_t *handle, nvgraphGraphDescr_t *graph) {
  int source_vert = 0;
  unsigned long long count = 0;
  unsigned long long next;
  for (unsigned int i = 0; i < 25; ++i) {
    auto start_time = std::chrono::high_resolution_clock::now();

    nvgraphSssp(*handle, *graph, 0, &source_vert, 0);
    cudaDeviceSynchronize();

    auto end_time = std::chrono::high_resolution_clock::now();
    if (i > 4) {
      next = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time -
                                                                  start_time)
                 .count();
      std::cout << "Iteration " << std::setw(2) << i - 4 << " took " << next
                << " nanoseconds\n";
      count += next;
    }
  }
  std::cout << "Average (";
  if (args.use_alternate_graph) {
    std::cout << "using larger alternate graph";
  } else {
    std::cout << "using regular graph";
  }
  std::cout << ") of 20 runs == " << count / 20 << " nanoseconds" << std::endl;
}

int main(int argc, char **argv) {
  Args args;
  if (args.ParseArgs(argc, argv)) {
    return 0;
  } else if (!args.enable_print_output && !args.enable_timings) {
    std::cout << "ERROR: Neither print-output nor timings were enabled"
              << std::endl;
    Args::DisplayHelp();
    return 1;
  }

  nvgraphHandle_t handle;
  nvgraphGraphDescr_t graph;
  nvgraphCSCTopology32I_st CSC_input;

  if (args.enable_timings) {
    if (args.use_alternate_graph) {
      SetUpGraphAlt(&handle, &graph, &CSC_input, args.randomize_weights,
                    args.seed_random_with_time);
    } else {
      SetUpGraph(&handle, &graph, &CSC_input, args.randomize_weights,
                 args.seed_random_with_time);
    }
    DoTimedRuns(args, CSC_input, &handle, &graph);
  } else /* if (args.enable_print_output) */ {
    DoPrintOutRun(args, CSC_input, &handle, &graph);
  }

  // cleanup
  CleanUpGraph(&handle, &graph, &CSC_input);

  return 0;
}
