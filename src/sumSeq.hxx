#pragma once
#include <vector>
#include "sum.hxx"

using std::vector;




template <class T>
SumResult<T> sumSeq(const T *x, size_t N, const SumOptions& o={}) {
  ASSERT(x);
  T a = T();
  float t = measureDuration([&]() {
    a = sumValues(x, N);
  }, o.repeat);
  return {a, t};
}

template <class T>
SumResult<T> sumSeq(const vector<T>& x, const SumOptions& o={}) {
  return sumSeq(x.data(), x.size(), o);
}
