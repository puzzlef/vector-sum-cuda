#pragma once
#include "_main.hxx"




struct SumOptions {
  int repeat;
  int gridLimit;
  int blockSize;
  int mode;

  SumOptions(int repeat=1, int gridLimit=GRID_LIMIT, int blockSize=BLOCK_LIMIT, int mode=0) :
  repeat(repeat), gridLimit(gridLimit), blockSize(blockSize), mode(mode) {}
};

template <class T>
struct SumResult {
  T   result;
  float time;

  SumResult(T result, float time=0) :
  result(result), time(time) {}
};
