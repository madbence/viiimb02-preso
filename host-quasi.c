#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

int main(int argc, char *argv[])
{
  size_t n = 1000;
  size_t i;
  curandGenerator_t gen;
  float *devData, *hostData;
  int r[1000] = {0};

  hostData = (float *)calloc(n, sizeof(float));
  CUDA_CALL(cudaMalloc((void **)&devData, n*sizeof(float)));
  CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_QUASI_DEFAULT));
  CURAND_CALL(curandSetQuasiRandomGeneratorDimensions(gen, 2));
  CURAND_CALL(curandGenerateUniform(gen, devData, n));
  CUDA_CALL(cudaMemcpy(hostData, devData, n * sizeof(float), cudaMemcpyDeviceToHost));

  for(i = 0; i < n; i++) {
    r[(int)(hostData[i] * 100)]++;
  }
  printf("x,y,label\n");
  for(i = 0; i < n / 2; i++) {
    printf("%lg,%lg,a\n", hostData[i] * 100, hostData[i + n / 2] * 100);
  }

  CURAND_CALL(curandDestroyGenerator(gen));
  CUDA_CALL(cudaFree(devData));
  free(hostData);
  return EXIT_SUCCESS;
}
