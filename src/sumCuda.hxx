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
    if (i < N) a[i] += a[N+i];
    __syncthreads();
  }
}


template <class T, int S, int R>
__device__ T sumKernelLoop(T *x, int N, int i, int DI) {
  T a = T();
  for (; i+(R-1)*S<N; i+=DI) {
    switch (R) {
      default:
      case 1: a += x[i+0*S]; break;
      case 2: a += x[i+0*S] + x[i+1*S]; break;
      case 4: a += x[i+0*S] + x[i+1*S] + x[i+2*S] + x[i+3*S]; break;
    }
  }
  switch (R) {
    case 2:
      if (i+0*S<N) a += x[i+0*S];
      break;
    case 4:
      if (i+0*S<N) a += x[i+0*S];
      if (i+1*S<N) a += x[i+1*S];
      if (i+2*S<N) a += x[i+2*S];
      break;
  }
  return a;
}


template <class T, int S, int R=1>
__global__ void sumKernel(T *a, T *x, int N) {
  DEFINE(t, b, B, G);
  __shared__ T cache[BLOCK_LIMIT];

  cache[t] = sumKernelLoop<T, S, R>(x, N, R*B*b+t, G*R*B);
  sumKernelReduce(cache, B, t);
  if (t==0) a[b] = cache[0];
}


template <class T>
void sumKernelCu(T *a, T *x, int N, int G, int B, int R=1) {
  switch (B) {
    default:
      printf("sumKernelCu: Bad block size %d!\n", B);
      break;
    case 32:
      switch (R) {
        default:
        case 1: sumKernel<T, 32, 1><<<G, B>>>(a, x, N); break;
        case 2: sumKernel<T, 32, 2><<<G, B>>>(a, x, N); break;
        case 4: sumKernel<T, 32, 4><<<G, B>>>(a, x, N); break;
      }
      break;
    case 64:
      switch (R) {
        default:
        case 1: sumKernel<T, 64, 1><<<G, B>>>(a, x, N); break;
        case 2: sumKernel<T, 64, 2><<<G, B>>>(a, x, N); break;
        case 4: sumKernel<T, 64, 4><<<G, B>>>(a, x, N); break;
      }
      break;
    case 128:
      switch (R) {
        default:
        case 1: sumKernel<T, 128, 1><<<G, B>>>(a, x, N); break;
        case 2: sumKernel<T, 128, 2><<<G, B>>>(a, x, N); break;
        case 4: sumKernel<T, 128, 4><<<G, B>>>(a, x, N); break;
      }
      break;
    case 256:
      switch (R) {
        default:
        case 1: sumKernel<T, 256, 1><<<G, B>>>(a, x, N); break;
        case 2: sumKernel<T, 256, 2><<<G, B>>>(a, x, N); break;
        case 4: sumKernel<T, 256, 4><<<G, B>>>(a, x, N); break;
      }
      break;
    case 512:
      switch (R) {
        default:
        case 1: sumKernel<T, 512, 1><<<G, B>>>(a, x, N); break;
        case 2: sumKernel<T, 512, 2><<<G, B>>>(a, x, N); break;
        case 4: sumKernel<T, 512, 4><<<G, B>>>(a, x, N); break;
      }
      break;
    case 1024:
      switch (R) {
        default:
        case 1: sumKernel<T, 1024, 1><<<G, B>>>(a, x, N); break;
        case 2: sumKernel<T, 1024, 2><<<G, B>>>(a, x, N); break;
        case 4: sumKernel<T, 1024, 4><<<G, B>>>(a, x, N); break;
      }
      break;
  }
}


template <class T>
SumResult<T> sumCuda(const T *x, int N, const SumOptions& o={}) {
  int R = 1 << o.mode;
  int B = o.blockSize;
  int G = max(prevPow2(min(ceilDiv(N, R*B), o.gridLimit)), 32);
  int C = 128; // decent sum threads
  int H = max(prevPow2(min(ceilDiv(G, R*C), BLOCK_LIMIT)), 32);
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
      sumKernelCu(bD, xD, N, G, B, R);
      sumKernelCu(aD, bD, G, H, C, R);
      sumKernelCu(aD, aD, H, 1, H, R);
    }
    else {
      sumKernelCu(aD, xD, N, G, B, R);
      sumKernelCu(aD, aD, G, 1, G, R);
    }
    TRY( cudaDeviceSynchronize() );
  }, o.repeat);
  TRY( cudaMemcpy(&a, aD, sizeof(T), cudaMemcpyDeviceToHost) );

  TRY( cudaFree(aD) );
  TRY( cudaFree(bD) );
  TRY( cudaFree(xD) );
  return {a, t};
}

template <class T>
SumResult<T> sumCuda(const vector<T>& x, const SumOptions& o={}) {
  return sumCuda(x.data(), x.size(), o);
}
