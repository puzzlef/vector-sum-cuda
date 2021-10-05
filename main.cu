#include <vector>
#include <string>
#include <cstdio>
#include "src/main.hxx"

using namespace std;




#define TYPE double


void runSum(int N, int repeat) {
  vector<TYPE> x(N);
  for (int i=0; i<N; i++)
    x[i] = 1.0/(i+1);

  // Find Σx using a single thread.
  auto a1 = sumSeq(x, {repeat});
  printf("[%09.3f ms] [%f] sumSeq\n", a1.time, a1.result);

  // Find Σx using memcpy based CUDA.
  auto a2 = sumMemcpyCuda(x, {repeat});
  printf("[%09.3f ms] [%f] sumMemcpyCuda\n", a2.time, a2.result);

  // Find Σx using in-place based CUDA.
  auto a3 = sumInplaceCuda(x, {repeat});
  printf("[%09.3f ms] [%f] sumInplaceCuda\n", a3.time, a3.result);
}


int main(int argc, char **argv) {
  int repeat = argc>1? stoi(argv[1]) : 5;
  for (int n=1000000; n<=1000000000; n*=10) {
    printf("# Elements %.0e\n", (double) n);
    runSum(n, repeat);
    printf("\n");
  }
  return 0;
}
