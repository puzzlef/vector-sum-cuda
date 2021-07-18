#include <vector>
#include <string>
#include <cstdio>
#include "src/main.hxx"

using namespace std;




void runSum(int N, int repeat) {
  vector<double> x(N);
  for (int i=0; i<N; i++)
    x[i] = 1.0/(i+1);

  // Find Σx using a single thread.
  auto a1 = sumSeq(x, {repeat});
  printf("[%09.3f ms] [%f] sumSeq\n", a1.time, a1.result);

  // Find Σx accelerated using CUDA.
  for (int grid=1024; grid<=GRID_LIMIT; grid*=2) {
    for (int block=32; block<=BLOCK_LIMIT; block*=2) {
      auto a2 = sumCuda(x, {repeat, grid, block});
      printf("[%09.3f ms] [%f] sumCuda<<<%d, %d>>>\n", a2.time, a2.result, grid, block);
    }
  }
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
