#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
  printf("Error at %s:%d\n",__FILE__,__LINE__);\
  return EXIT_FAILURE;}} while(0)

__global__ void setup(curandState *state) {
  int id = threadIdx.x + blockIdx.x * 64;
  curand_init(1234, id, 0, &state[id]);
}

__global__ void generate(curandState *state, int n, unsigned int* result) {
  int id = threadIdx.x + blockIdx.x * 64;
  float x;
  curandState localState = state[id];
  for(int i = 0; i < n; i++) {
    x = curand_uniform(&localState);
    result[i * 100 + (int)(x*100)]++;
  }
  state[id] = localState;
}

int main(int argc, char *argv[]) {
  int i, j;
  curandState *devStates;
  unsigned int *devResults, *hostResults;
  int samples = 10000;
  unsigned int r[100] = {0};

  hostResults = (unsigned int *)calloc(64 * 64 * 100, sizeof(int));
  CUDA_CALL(cudaMalloc((void **)&devResults, 100 * 64 * 64 * sizeof(unsigned int)));
  CUDA_CALL(cudaMemset(devResults, 0, 100 * 64 * 64 * sizeof(unsigned int)));
  CUDA_CALL(cudaMalloc((void **)&devStates, 64 * 64 * sizeof(curandState)));

  setup<<<64, 64>>>(devStates);
  generate<<<64, 64>>>(devStates, samples, devResults);

  CUDA_CALL(cudaMemcpy(hostResults, devResults, 100 * 64 * 64 * sizeof(unsigned int), cudaMemcpyDeviceToHost));

  for (i = 0; i < 64 * 64; i++) {
    for (j = 0; j < 100; j++) {
      r[j] += hostResults[i * 100 + j];
    }
  }

  printf("x,y\n");
  for (i = 0; i < 100; i++) {
    printf("%d,%d\n", i, r[i]);
  }
}
