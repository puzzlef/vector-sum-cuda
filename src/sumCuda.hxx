#pragma once
#include <vector>
#include <algorithm>
#include "_main.hxx"
#include "sum.hxx"
#include "sumSeq.hxx"

using std::vector;
using std::min;




template <class T, int S>
__device__ void sumKernelReduceWarp(volatile T* a, int i) {
  if (S>32) a[i] += a[i+32];
  if (S>16) a[i] += a[i+16];
  if (S>8)  a[i] += a[i+8];
  if (S>4)  a[i] += a[i+4];
  if (S>2)  a[i] += a[i+2];
  if (S>1)  a[i] += a[i+1];
}

template <class T, int S>
__device__ void sumKernelReduce(T* a, int i) {
  __syncthreads();
  if (S>512) { if (i<512) a[i] += a[i+512]; __syncthreads(); }
  if (S>256) { if (i<256) a[i] += a[i+256]; __syncthreads(); }
  if (S>128) { if (i<128) a[i] += a[i+128]; __syncthreads(); }
  if (S>64)  { if (i<64)  a[i] += a[i+64];  __syncthreads(); }
  if (i<32) sumKernelReduceWarp<T, S>(a, i);
}


template <class T>
__device__ T sumKernelLoop(T *x, int N, int i, int DI) {
  T a = T();
  for (; i<N; i+=DI)
    a += x[i];
  return a;
}


template <class T, int S>
__global__ void sumKernel(T *a, T *x, int N) {
  DEFINE(t, b, B, G);
  __shared__ T cache[BLOCK_LIMIT];

  cache[t] = sumKernelLoop(x, N, B*b+t, G*B);
  sumKernelReduce<T, S>(cache, t);
  if (t==0) a[b] = cache[0];
}


template <class T>
void sumKernelCu(T *a, T *x, int N, int G, int B) {
  switch (B) {
    default: printf("sumKernelCu: Bad block size %d!\n", B); break;
    case   32: sumKernel<T,   32><<<G, B>>>(a, x, N); break;
    case   64: sumKernel<T,   64><<<G, B>>>(a, x, N); break;
    case  128: sumKernel<T,  128><<<G, B>>>(a, x, N); break;
    case  256: sumKernel<T,  256><<<G, B>>>(a, x, N); break;
    case  512: sumKernel<T,  512><<<G, B>>>(a, x, N); break;
    case 1024: sumKernel<T, 1024><<<G, B>>>(a, x, N); break;
  }
}


template <class T>
SumResult<T> sumCuda(const T *x, int N, const SumOptions& o={}) {
  int B = o.blockSize;
  int G = max(prevPow2(min(ceilDiv(N, B), o.gridLimit)), 32);
  int C = 128; // decent sum threads
  int H = max(prevPow2(min(ceilDiv(G, C), BLOCK_LIMIT)), 32);
  size_t N1 = N * sizeof(T);
  size_t G1 = G * sizeof(T);

  T *aD, *bD, *xD;
  TRY( cudaMalloc(&aD, G1) );
  TRY( cudaMalloc(&bD, G1) );
  TRY( cudaMalloc(&xD, N1) );
  TRY( cudaMemcpy(xD, x, N1, cudaMemcpyHostToDevice) );

  T a = T();
  float t = measureDuration([&] {
    if (G>BLOCK_LIMIT) {
      sumKernelCu(bD, xD, N, G, B);
      sumKernelCu(aD, bD, G, H, C);
      sumKernelCu(aD, aD, H, 1, H);
    }
    else {
      sumKernelCu(aD, xD, N, G, B);
      sumKernelCu(aD, aD, G, 1, G);
    }
    TRY( cudaDeviceSynchronize() );
  }, o.repeat);
  TRY( cudaMemcpy(&a, aD, sizeof(T), cudaMemcpyDeviceToHost) );

  TRY( cudaFree(aD) );
  TRY( cudaFree(xD) );
  return {a, t};
}

template <class T>
SumResult<T> sumCuda(const vector<T>& x, const SumOptions& o={}) {
  return sumCuda(x.data(), x.size(), o);
}
