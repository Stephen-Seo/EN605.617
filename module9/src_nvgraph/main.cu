#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include <nvgraph.h>

int main(int argc, char **argv) {
  nvgraphHandle_t handle;
  nvgraphGraphDescr_t graph;
  nvgraphCSCTopology32I_st CSC_input;
  CSC_input.nvertices = 8;
  CSC_input.nedges = 14;
  CSC_input.destination_offsets =
      (int *)malloc(sizeof(int) * (CSC_input.nvertices + 1));
  CSC_input.destination_offsets[CSC_input.nvertices] = CSC_input.nedges;
  CSC_input.source_indices = (int *)malloc(sizeof(int) * CSC_input.nedges);
  float *edge_weights = (float *)malloc(sizeof(float) * CSC_input.nedges);

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

  CSC_input.destination_offsets[0] = 0;
  CSC_input.destination_offsets[1] = 1;
  CSC_input.destination_offsets[2] = 2;
  CSC_input.destination_offsets[3] = 3;
  CSC_input.destination_offsets[4] = 4;
  CSC_input.destination_offsets[5] = 6;
  CSC_input.destination_offsets[6] = 9;
  CSC_input.destination_offsets[7] = 11;
  // index 8 is already set earlier

  CSC_input.source_indices[0] = 7;
  CSC_input.source_indices[1] = 0;
  CSC_input.source_indices[2] = 0;
  CSC_input.source_indices[3] = 0;
  CSC_input.source_indices[4] = 1;
  CSC_input.source_indices[5] = 2;
  CSC_input.source_indices[6] = 1;
  CSC_input.source_indices[7] = 2;
  CSC_input.source_indices[8] = 3;
  CSC_input.source_indices[9] = 2;
  CSC_input.source_indices[10] = 3;
  CSC_input.source_indices[11] = 4;
  CSC_input.source_indices[12] = 5;
  CSC_input.source_indices[13] = 6;

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

  nvgraphCreate(&handle);

  nvgraphCreate(&handle);
  nvgraphCreateGraphDescr(handle, &graph);

  nvgraphSetGraphStructure(handle, graph, &CSC_input, NVGRAPH_CSC_32);

  // Only one set of Vertex data will hold the result
  nvgraphAllocateVertexData(handle, graph, 1, &set_type);

  // Only one set of Edge data will hold the weights
  nvgraphAllocateEdgeData(handle, graph, 1, &set_type);
  nvgraphSetEdgeData(handle, graph, edge_weights, 0);

  // Check shortest path starting at 0
  int source_vert = 0;
  nvgraphSssp(handle, graph, 0, &source_vert, 0);

  // Get result data from vertex set
  std::vector<float> output_data(CSC_input.nvertices);
  nvgraphGetVertexData(handle, graph, output_data.data(), 0);

  std::cout << "Got result values:\n";
  for (unsigned int i = 0; i < CSC_input.nvertices; ++i) {
    std::cout << std::setw(2) << i << ": " << std::setw(5) << std::fixed
              << std::setprecision(2) << output_data[i] << std::endl;
  }

  // cleanup
  nvgraphDestroyGraphDescr(handle, graph);
  nvgraphDestroy(handle);

  free(CSC_input.source_indices);
  free(CSC_input.destination_offsets);
  free(edge_weights);

  return 0;
}
