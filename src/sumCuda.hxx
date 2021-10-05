#pragma once
#include <vector>
#include <algorithm>
#include "_main.hxx"
#include "sum.hxx"

using std::vector;
using std::min;




template <class T>
__device__ void sumKernelReduce(T* a, int N, int i) {
  __syncthreads();
  for (N=N/2; N>0; N/=2) {
    if (i < N) a[i] += a[N+i];
    __syncthreads();
  }
}


template <class T>
__device__ T sumKernelLoop(T *x, int N, int i, int DI) {
  T a = T();
  for (; i<N; i+=DI)
    a += x[i];
  return a;
}


template <class T>
__global__ void sumKernel(T *a, T *x, int N) {
  DEFINE(t, b, B, G);
  __shared__ T cache[BLOCK_LIMIT];
  cache[t] = sumKernelLoop(x, N, B*b+t, G*B);
  sumKernelReduce(cache, B, t);
  if (t == 0) a[b] = cache[0];
}


template <class T>
SumResult<T> sumCuda(const T *x, int N, const SumOptions& o={}) {
  int B = o.blockSize;
  int G = min(ceilDiv(N, B), o.gridLimit);
  size_t N1 = N * sizeof(T);
  size_t G1 = G * sizeof(T);

  T *aD, *xD, aH[GRID_LIMIT];
  TRY( cudaMalloc(&aD, G1) );
  TRY( cudaMalloc(&xD, N1) );
  TRY( cudaMemcpy(xD, x, N1, cudaMemcpyHostToDevice) );

  T a = T();
  float t = measureDuration([&] {
    sumKernel<<<G, B>>>(aD, xD, N);
    TRY( cudaMemcpy(aH, aD, G1, cudaMemcpyDeviceToHost) );
    a = sum(aH, G);
  }, o.repeat);

  TRY( cudaFree(aD) );
  TRY( cudaFree(xD) );
  return {a, t};
}

template <class T>
SumResult<T> sumCuda(const vector<T>& x, const SumOptions& o={}) {
  return sumCuda(x.data(), x.size(), o);
}
