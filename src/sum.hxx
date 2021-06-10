#pragma once
#include "_main.hxx"




struct SumOptions {
  int repeat;
  int gridLimit;
  int blockSize;

  SumOptions(int repeat=1, int gridLimit=GRID_LIMIT, int blockSize=BLOCK_LIMIT) :
  repeat(repeat), gridLimit(gridLimit), blockSize(blockSize) {}
};

template <class T>
struct SumResult {
  T   result;
  float time;

  SumResult(T result, float time=0) :
  result(result), time(time) {}
};
