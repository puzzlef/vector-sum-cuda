#pragma once
#include <vector>
#include "_main.hxx"
#include "sum.hxx"

using std::vector;




template <bool POW2=false, class T>
SumResult<T> sumCuda(const T *x, int N, const SumOptions& o={}) {
  ASSERT(x);
  int RN = reduceSizeCu(N);
  size_t RT1 = RN * sizeof(T);
  size_t NT1 = N  * sizeof(T);
  ASSERT(RN<=N);
  T a   = T();
  T *r  = nullptr;
  T *rD = nullptr;
  T *xD = nullptr;
  // TRY( cudaProfilerStart() );
  TRY( cudaSetDeviceFlags(cudaDeviceMapHost) );
  TRY( cudaHostAlloc(&r, RT1, cudaHostAllocDefault) );
  TRY( cudaMalloc(&rD, RT1) );
  TRY( cudaMalloc(&xD, NT1) );
  TRY( cudaMemcpy(xD, x, NT1, cudaMemcpyHostToDevice) );
  float t = measureDuration([&] {
    sumCuW<POW2>(rD, xD, N);
    TRY( cudaMemcpy(r, rD, RT1, cudaMemcpyDeviceToHost) );
    a = sumValues(r, RN);
  }, o.repeat);
  TRY( cudaFreeHost(r) );
  TRY( cudaFree(rD) );
  TRY( cudaFree(xD) );
  // TRY( cudaProfilerStart() );
  return {a, t};
}

template <bool POW2=false, class T>
SumResult<T> sumCuda(const vector<T>& x, const SumOptions& o={}) {
  return sumCuda<POW2>(x.data(), x.size(), o);
}
