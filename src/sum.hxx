#pragma once
#include "_main.hxx"




struct SumOptions {
  int repeat;
  int blockSize;
  int threadDuty;

  SumOptions(int repeat=1, int blockSize=BLOCK_LIMIT, int threadDuty=1) :
  repeat(repeat), blockSize(blockSize), threadDuty(threadDuty) {}
};

template <class T>
struct SumResult {
  T   result;
  float time;

  SumResult(T result, float time=0) :
  result(result), time(time) {}
};
