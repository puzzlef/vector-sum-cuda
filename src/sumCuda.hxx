#pragma once
#include <vector>
#include <algorithm>
#include "_main.hxx"
#include "sum.hxx"
#include "sumSeq.hxx"

using std::vector;
using std::min;




template <class T>
__device__ void sumKernelReduce(T* a, int N, int i) {
  __syncthreads();
  for (N=N/2; N>0; N/=2) {
    if (i<N) a[i] += a[N+i];
    __syncthreads();
  }
}


template <class T>
__device__ T sumKernelLoop(const T *x, int N, int i, int DI) {
  T a = T();
  for (; i<N; i+=DI)
    a += x[i];
  return a;
}


template <class T, int C=BLOCK_LIMIT>
__global__ void sumKernel(T *a, const T *x, int N) {
  DEFINE(t, b, B, G);
  __shared__ T cache[C];

  cache[t] = sumKernelLoop(x, N, B*b+t, G*B);
  sumKernelReduce(cache, B, t);
  if (t==0) a[b] = cache[0];
}


template <class T>
void sumMemcpyCu(T *a, const T *x, int N) {
  const int B = BLOCK_DIM_R<T>();
  const int G = min(ceilDiv(N, B), GRID_DIM_R<T>());
  sumKernel<<<G, B>>>(a, x, N);
}

template <class T>
void sumInplaceCu(T *a, const T *x, int N) {
  const int B = BLOCK_DIM_R<T>();
  const int G = min(ceilDiv(N, B), GRID_DIM_R<T>());
  sumKernel<<<G, B>>>(a, x, N);
  sumKernel<<<1, G>>>(a, a, G);
}

template <class T>
void sumCu(T *a, const T *x, int N) {
  sumInplaceCu(a, x, N);
}




template <class T>
SumResult<T> sumMemcpyCuda(const T *x, int N, const SumOptions& o={}) {
  const int G = GRID_DIM_R<T>();
  size_t N1 = N * sizeof(T);
  size_t G1 = G * sizeof(T);

  T *aD, *xD, aH[G];
  TRY( cudaMalloc(&aD, G1) );
  TRY( cudaMalloc(&xD, N1) );
  TRY( cudaMemcpy(xD, x, N1, cudaMemcpyHostToDevice) );

  T a = T();
  float t = measureDuration([&] {
    sumMemcpyCu(aD, xD, N);
    TRY( cudaMemcpy(aH, aD, G1, cudaMemcpyDeviceToHost) );
    a = sumLoop(aH, reduceSizeCu<T>(N));
  }, o.repeat);

  TRY( cudaFree(aD) );
  TRY( cudaFree(xD) );
  return {a, t};
}

template <class T>
SumResult<T> sumMemcpyCuda(const vector<T>& x, const SumOptions& o={}) {
  return sumMemcpyCuda(x.data(), x.size(), o);
}




template <class T>
SumResult<T> sumInplaceCuda(const T *x, int N, const SumOptions& o={}) {
  const int G = GRID_DIM_R<T>();
  size_t N1 = N * sizeof(T);
  size_t G1 = G * sizeof(T);

  T *aD, *xD;
  TRY( cudaMalloc(&aD, G1) );
  TRY( cudaMalloc(&xD, N1) );
  TRY( cudaMemcpy(xD, x, N1, cudaMemcpyHostToDevice) );

  T a = T();
  float t = measureDuration([&] {
    sumInplaceCu(aD, xD, N);
    TRY( cudaDeviceSynchronize() );
  }, o.repeat);
  TRY( cudaMemcpy(&a, aD, sizeof(T), cudaMemcpyDeviceToHost) );

  TRY( cudaFree(aD) );
  TRY( cudaFree(xD) );
  return {a, t};
}

template <class T>
SumResult<T> sumInplaceCuda(const vector<T>& x, const SumOptions& o={}) {
  return sumInplaceCuda(x.data(), x.size(), o);
}
